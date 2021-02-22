import grpc
import digestor_pb2
import digestor_pb2_grpc
 
class DigestorClient(object):
    """
    Client for accessing the gRPC functionality
    """
 
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 46001

        self.channel = grpc.insecure_channel(
                        '{}:{}'.format(self.host, self.server_port))
 

        self.stub = digestor_pb2_grpc.DigestorStub(self.channel)
 
    def get_digest(self, message):
        """
        Client function to call the rpc for GetDigest
        """
        to_digest_message =digestor_pb2.DigestMessage(ToDigest=message)
        return self.stub.GetDigestor(to_digest_message)