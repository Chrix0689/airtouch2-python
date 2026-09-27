import asyncio
import errno
import logging
import socket
from typing import Callable
from airtouch2.common.interfaces import CoroCallback, Serializable, TaskCreator

_LOGGER = logging.getLogger(__name__)

NetworkOrHostDownErrors = (errno.EHOSTUNREACH, errno.ECONNREFUSED,  errno.ETIMEDOUT,
                           errno.ENETDOWN, errno.ENETUNREACH, errno.ENETRESET, errno.ECONNABORTED)

def _set_keepalive_options(
    sock: socket.socket, idle_seconds: int, interval_seconds: int, count: int
):
    if hasattr(sock, "SO_KEEPALIVE"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if hasattr(sock, "TCP_KEEPIDLE"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, idle_seconds)
    if hasattr(socket, "TCP_KEEPINTVL"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_seconds)
    if hasattr(socket, "TCP_KEEPCNT"):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, count)
    if hasattr(socket, "TCP_USER_TIMEOUT"):
        sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_USER_TIMEOUT,
            1000 * (idle_seconds + (interval_seconds * count)),
        )

class NetClient:
    """A generic network client"""

    def __init__(self, host: str, port: int, on_connect: CoroCallback, handle_message: CoroCallback,
                 task_creator: TaskCreator = asyncio.create_task):
        # network
        self._host_ip: str = host
        self._host_port: int = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

        # async
        self._task_creator: Callable = task_creator
        self._main_loop_task: asyncio.Task[None] | None = None
        self._stop: bool = False
        self._reconnect_lock = asyncio.Lock()

        self._on_connect = on_connect
        self._handle_message = handle_message

    async def connect(self) -> bool:
        """Opens connection to the server, returns True/False if successful/unsuccessful"""
        _LOGGER.debug(f"Connecting to {self._host_ip} on port {self._host_port}")
        try:
            self._reader, self._writer = await asyncio.open_connection(self._host_ip, self._host_port)
        except OSError as e:
            _LOGGER.warning(f"Could not connect to host {self._host_ip}")
            if isinstance(e, socket.gaierror):
                # provided ip or port is rubbish/invalid
                pass
            elif e.errno not in NetworkOrHostDownErrors:
                raise e
            return False
        else:
            if self._stop:
                self._close_connection()
                return False
            _set_keepalive_options(
                self._writer.get_extra_info("socket"),
                idle_seconds=5,
                interval_seconds=1,
                count=5,
            )
            await self._on_connect()
            return True

    def run(self) -> None:
        """Starts the processing of incoming information from the server"""
        _LOGGER.debug("Starting listener task")
        self._main_loop_task = self._task_creator(self._main())

    async def stop(self) -> None:
        """Stops the processing of incoming information from the server"""
        self._stop = True
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
            finally:
                self._main_loop_task = None
                self._close_connection()
        else:
            self._close_connection()

    def _close_connection(self):
        if self._writer is not None:
            self._writer.close()
        self._reader = None
        self._writer = None

    async def reconnect(self):
        """Replace an unresponsive connection, sharing concurrent recovery."""
        await self._try_reconnect(self._writer)

    async def send(self, message: Serializable, *, retry_safe: bool = False) -> None:
        """Send a message; replay only explicitly idempotent messages once.

        A failed drain does not tell us whether the controller applied a command.
        Relative commands must surface that uncertainty, never be replayed.
        """
        if self._writer is None and retry_safe and not self._stop:
            await asyncio.wait_for(self._try_reconnect(None), timeout=10)
        if self._writer is None or self._stop:
            raise RuntimeError("Client is not connected - call connect() first")
        else:
            bytes_to_write = message.to_bytes()
            _LOGGER.debug(f"Sending {message.__class__.__name__} with data: {bytes_to_write.hex(':')}")
            _LOGGER.debug(f"{repr(message)}")
            for attempt in range(2):
                writer = self._writer
                try:
                    if writer is None:
                        raise ConnectionError("AirTouch connection is recovering")
                    writer.write(bytes_to_write)
                    await writer.drain()
                    return
                except (OSError, asyncio.IncompleteReadError) as e:
                    if not retry_safe or attempt:
                        raise ConnectionError(
                            "AirTouch command delivery could not be confirmed"
                        ) from e
                    await asyncio.wait_for(self._try_reconnect(writer), timeout=10)

    async def read_bytes(self, size: int) -> bytes | None:
        """
        Read exactly 'size' bytes, return None if could not read enough bytes or on disconnection and reconnection.
        This coroutine handles reconnection.
        """
        if self._reader is None:
            await self._try_reconnect(self._writer)
            return None
        reader, writer = self._reader, self._writer
        try:
            data = await reader.readexactly(size)
        except asyncio.IncompleteReadError as e:
            _LOGGER.debug(f"IncompleteReadError - partial bytes: {e.partial.hex(':')}")
            data = None
        except OSError as e:
            _LOGGER.debug("ConnectionResetError")
            data = None

        if data is None:
            _LOGGER.warning("Connection lost, reconnecting")
            await self._try_reconnect(writer)
            return None
        if reader is not self._reader:
            return None
        _LOGGER.debug(f"Read payload of size {size}: {data.hex(':')}")
        return data

    async def _main(self) -> None:
        while not self._stop:
            await self._handle_message()

    async def _try_reconnect(self, failed_writer) -> None:
        async with self._reconnect_lock:
            if self._stop:
                raise ConnectionError("AirTouch client has stopped")
            if self._writer is not None and self._writer is not failed_writer:
                return
            self._close_connection()
            retries = 0
            while not self._stop:
                try:
                    if await self.connect():
                        _LOGGER.info("Reconnected")
                        return
                except OSError:
                    pass
                except BaseException:
                    self._close_connection()
                    raise
                self._close_connection()
                await asyncio.sleep(min(0.1 * 2 ** min(retries, 7), 10))
                retries += 1
            raise ConnectionError("AirTouch client has stopped")
