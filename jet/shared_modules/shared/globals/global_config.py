from typing import Dict

# Define the type for the headers dictionary
HeadersType = Dict[str, str]

# Define the type for the settings dictionary
SettingsType = Dict[str, Dict[str, HeadersType]]

# The actual headers data
headers: HeadersType = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzZXNzaW9uSWQiOiI4ZjQ4Y2Q0ZS1iYjU4LTQ1ZmMtYWVkMC1mMDMxMDM5NTczNzEiLCJpYXQiOjE3Mzg3NDE5MzEsImV4cCI6MTczODgyODMzMX0.-mPmlF8Z0H9WAcFIs8TIu_mfoer_RFeHpQ7awdxZkVk",
}

# The actual settings data
settings: SettingsType = {
    "request": {
        "headers": headers
    }
}

__all__ = [
    "settings",
]
