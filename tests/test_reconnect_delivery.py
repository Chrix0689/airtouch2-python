"""No hardware is used: failed sockets reproduce the lost first command."""
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from airtouch2.common.NetClient import NetClient
from airtouch2.at2plus import At2PlusClient
from airtouch2.protocol.at2plus.enums import AcSetPower, AcSetMode, AcFanSpeed, GroupSetPower, GroupSetDamper
from airtouch2.protocol.at2plus.messages.AcControl import AcControlMessage, AcSettings
from airtouch2.protocol.at2plus.messages.GroupControl import GroupControlMessage, GroupSettings
from airtouch2.protocol.at2plus.messages.AcStatus import AcStatusMessage


class TestReconnectDelivery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = NetClient("unused", 9200, AsyncMock(), AsyncMock())
        self.old = Mock()
        self.old.drain = AsyncMock(side_effect=ConnectionResetError())
        self.new = Mock()
        self.new.drain = AsyncMock()
        self.client._writer = self.old
        async def reconnect(failed_writer):
            self.client._writer = self.new
        self.client._try_reconnect = AsyncMock(side_effect=reconnect)
        self.message = AcStatusMessage([])

    async def test_safe_request_is_written_to_new_socket(self):
        await self.client.send(self.message, retry_safe=True)
        self.new.write.assert_called_once_with(self.message.to_bytes())

    async def test_write_failure_is_also_recovered(self):
        self.old.write.side_effect = BrokenPipeError()
        await self.client.send(self.message, retry_safe=True)
        self.new.write.assert_called_once_with(self.message.to_bytes())

    async def test_second_failure_is_reported_not_looped_forever(self):
        self.new.drain.side_effect = ConnectionResetError()
        with self.assertRaises(ConnectionError):
            await self.client.send(self.message, retry_safe=True)
        self.client._try_reconnect.assert_awaited_once()

    async def test_unsafe_delivery_is_not_silently_successful_or_replayed(self):
        with self.assertRaises(ConnectionError):
            await self.client.send(self.message)
        self.new.write.assert_not_called()
        self.client._try_reconnect.assert_not_awaited()

    async def test_cancellation_is_propagated(self):
        self.old.drain.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.client.send(self.message, retry_safe=True)
        self.client._try_reconnect.assert_not_awaited()

    async def test_at2plus_only_opts_in_absolute_commands(self):
        client = At2PlusClient("unused")
        client._client.send = AsyncMock()
        for power, expected in ((AcSetPower.ON, True), (AcSetPower.OFF, True), (AcSetPower.TOGGLE, False)):
            message = AcControlMessage([AcSettings(0, power, AcSetMode.UNCHANGED, AcFanSpeed.UNCHANGED)])
            await client.send(message)
            client._client.send.assert_awaited_with(message, retry_safe=expected)
        for power, damp_mode, expected in (
            (GroupSetPower.ON, GroupSetDamper.SET, True),
            (GroupSetPower.OFF, GroupSetDamper.UNCHANGED, True),
            (GroupSetPower.NEXT, GroupSetDamper.UNCHANGED, False),
            (GroupSetPower.UNCHANGED, GroupSetDamper.INC, False),
            (GroupSetPower.UNCHANGED, GroupSetDamper.DEC, False),
        ):
            message = GroupControlMessage([GroupSettings(0, damp_mode, power)])
            await client.send(message)
            client._client.send.assert_awaited_with(message, retry_safe=expected)


class TestConnectionLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_recovery_opens_only_one_connection(self):
        client = NetClient("unused", 9200, AsyncMock(), AsyncMock())
        old = Mock()
        new = Mock()
        client._writer = old
        started, release = asyncio.Event(), asyncio.Event()
        async def connect():
            started.set()
            await release.wait()
            client._reader, client._writer = Mock(), new
            return True
        client.connect = AsyncMock(side_effect=connect)
        first = asyncio.create_task(client._try_reconnect(old))
        await started.wait()
        second = asyncio.create_task(client._try_reconnect(old))
        release.set()
        await asyncio.gather(first, second)
        client.connect.assert_awaited_once()
        old.close.assert_called_once()
        self.assertIs(client._writer, new)
        await client.stop()
        new.close.assert_called_once()

    async def test_cancelled_handshake_closes_new_socket(self):
        client = NetClient("unused", 9200, AsyncMock(side_effect=asyncio.CancelledError()), AsyncMock())
        old, new = Mock(), Mock()
        client._writer = old
        with patch("asyncio.open_connection", AsyncMock(return_value=(Mock(), new))):
            with self.assertRaises(asyncio.CancelledError):
                await client._try_reconnect(old)
        old.close.assert_called_once()
        new.close.assert_called_once()
        self.assertIsNone(client._writer)

    async def test_stop_without_listener_closes_connection(self):
        client = NetClient("unused", 9200, AsyncMock(), AsyncMock())
        writer = Mock()
        client._writer = writer
        await client.stop()
        await client.stop()
        writer.close.assert_called_once()
        with self.assertRaises(ConnectionError):
            await client.reconnect()


if __name__ == "__main__":
    unittest.main()
