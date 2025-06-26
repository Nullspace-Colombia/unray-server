""" 
    Server - server.py

    Server Class utilities definition to create a simple 
    TCP/IP communication. 
    =====================================================
    Classes 
    =====================================================
     - ClientHandler
     - ServerHandler
    ====================================================
    
    QUICK START GUIDE 
    =====================================================
        To check the 
    =====================================================
"""
import socket, sys
from typing import Any
import numpy as np


class mySocket(socket.socket):
    def __init__(self, family, type):
        super().__init__(socket.AF_INET, socket.SOCK_STREAM)
        

    def __reduce__(self):
        return (self.__class__, (socket.AF_INET, socket.SOCK_STREAM))


class ClientHandler():
    __BUFFER_DATA_SIZE = 256
    """
    Client Handler 
    
    """
    def __init__(self, ip = 'localhost', port = 9443): 
        self.connected = False 
        self.ip = ip 
        self.port = port 

    
        
        
        
    def set_socket(self, ip = False):

        self.socket = mySocket(socket.AF_INET, socket.SOCK_STREAM) 
        return self.socket

    def get_socket(self):
        return self.sock

    def set_port(self, new_port):  
        self.port = new_port
    
    def connect(self, c_sock): 
        
        server_address = (self.ip, self.port)
        print('[ CONNECTION ] connecting to {} port {}'.format(*server_address))

        count = 1
        group_count = 1
        
        while not self.connected:
            try:
                c_sock.connect(server_address)
                self.connected = True 
            except: 
                count += 1
                if count % 4 == 0: 
                    print("Trying to connect...")        
                    group_count += 1
                    if group_count % 5 == 0: 
                        
                        break 
            return c_sock

        if not self.connected: 
            print("Connection Timeout!")          
            return False              
        else:
            print('[ CONNECTION ] Connected with server!')
            self.connected = False # Restart for future connections 
            return True # If Connection is realized 
        
        

    def send(self, msg, a_sock,  __BUFFER_DATA_SIZE = 32):
        if not self.connected:
            assert "No server connection. Please check"
        a_sock.settimeout(1)
        try:
            a_sock.send(msg)
        except:
            pass

        
        
    
    def recv(self, expected_bytes, b_sock):
        count = 0
        res = b''
        nuevos_datos = b''
        b_sock.settimeout(1)
       
        try:
            res = b_sock.recv(expected_bytes)
        
            
        except socket.error as e:
            "If the socket doesn't receive any data"
            print(f"Error : {e}")
            res = np.array([-1], dtype=np.double)
            print("RES", res)
            res = res.tobytes()

        respuesta = np.frombuffer(res, dtype=np.double) 

        return respuesta 
        


    def close(self, c_sock): 
        print("Closing connection...")
        self.connected = False
        c_sock.close()
        print("Connection closed! Bye.")


"""
class ServerHandler():
    
    
#Server Handler 

    
    def __init__(self, IP = 'localhost', PORT = 10000, max_connections = 2):
        self._IP   = IP
        self._PORT = PORT 
        self._MAX_CONNECTIONS = max_connections

        self.callback = None 
        self.number_connections = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_address = (self._IP, self._PORT)

        self.sock.bind(self.server_address)
        self.sock.listen(10)
       
    def start(self, __BUFFER_DATA_SIZE = 256):
        print('[ SERVER ] Begining server on {}:{} | TCP/IP'.format(*self.server_address))
        while True:
            connection, client_address = self.sock.accept() # Wait for connections 
            # Connected Procedure 
            self.new_connection()
            print('[ NEW ] {}:{} connected! ({}, {})'.format(*client_address, self.number_connections, self._MAX_CONNECTIONS)) 

            try:
                while True:
                    data = connection.recv(__BUFFER_DATA_SIZE)
                    
                    # print('received {!r}'.format(data), end = " ")
                    #  print(f"converted to: {np.frombytes(data)}")
                    if data:
                        print(f'Action: {np.frombuffer(data, dtype = np.float16) * 2}')
                        
                        connection.sendall(data)
                    else:
                        print('no data from', client_address)
                        break
            except:
                print("[ WARN ] No callback config. Please use ServerHandler.set_callback(fcn)")


    def get_connections(self):
        return self.number_connections

    def new_connection(self):
        self.number_connections += 1

    def shutdown(self): 
        self.sock.close() 

    def set_callback(self, fcn = None): 
        self.callback = fcn 

    def test_connection(self): 
        pass 

"""

