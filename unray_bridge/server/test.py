import time
import jwt

ak = "94cab13a6222439d94c67418ffbba74b" # fill access key
sk = "97b70cb2c2544a218d3fbd8075247bc3" # fill secret key

def encode_jwt_token(ak, sk):
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800, # The valid time, in this example, represents the current time+1800s(30min)
        "nbf": int(time.time()) - 5 # The time when it starts to take effect, in this example, represents the current time minus 5s
    }
    token = jwt.encode(payload, sk, headers=headers)
    return token

authorization = encode_jwt_token(ak, sk)
print(authorization) # Printing the generated API_TOKEN