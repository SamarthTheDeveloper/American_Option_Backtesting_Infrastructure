import asyncio
import grpc
import test_pb2
import test_pb2_grpc

#Simple case
# channel = grpc.insecure_channel("127.0.0.1:50051")
# delivery = test_pb2_grpc.DeliveryStub(channel)
# response = delivery.Send(test_pb2.TextRequest(text="hello"))
# print(response.text)
#python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/__.proto

async def main():
    async with grpc.aio.insecure_channel("127.0.0.1:50051") as channel:
        service = test_pb2_grpc.DeliveryStub(channel)
        call = service.Chat()

        for i in range(5):
            msg = input("Give a message: ")
            await call.write(test_pb2.TextRequest(text=msg))
            reply = await call.read()
            print("got: ", reply.text)

asyncio.run(main())
