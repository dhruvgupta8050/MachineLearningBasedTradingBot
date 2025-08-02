from SmartApi import SmartConnect

# Replace with your Angel One SmartAPI credentials
API_KEY = "vuu "
CLIENT_CODE = "cfgb"
PASSWORD = "xxxx"
TOTP = "xxxx"  # If 2FA is enabled

def authenticate_smartapi():
    obj = SmartConnect(api_key=API_KEY)
    try:
        data = obj.generateSession(CLIENT_CODE, PASSWORD, TOTP)
        print("Authentication successful!")
        print("Access Token:", data['data']['access_token'])
        return obj
    except Exception as e:
        print("Authentication failed:", str(e))
        return None

if __name__ == "__main__":
    authenticate_smartapi()