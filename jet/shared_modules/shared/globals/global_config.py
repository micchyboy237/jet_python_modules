from typing import Dict

# Define the type for the headers dictionary
HeadersType = Dict[str, str]

# Define the type for the settings dictionary
SettingsType = Dict[str, Dict[str, HeadersType]]

# The actual headers data
headers: HeadersType = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzZXNzaW9uSWQiOiI4NmI4YzU0Yy1lZjE1LTQxNTQtOTdhOS02N2E0NmNmNWU0OTIiLCJpYXQiOjE3Mzg4Mzc4NjMsImV4cCI6MTczODkyNDI2M30.12PYPw_88UpvSqnuL3LJcFH_dlOqyv29sLoeoltKp70",
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
