import grpc
from concurrent import futures
from typing import Dict, List

class GRPCService:
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self.server = None
        
    async def start_server(self):
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        await self.server.add_insecure_port(f'{self.host}:{self.port}')
        await self.server.start()
        return self.server
    
    async def stop_server(self):
        if self.server:
            await self.server.stop(5)
