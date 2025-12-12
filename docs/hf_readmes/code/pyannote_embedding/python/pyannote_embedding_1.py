# 1. visit hf.co/pyannote/embedding and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained model
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="ACCESS_TOKEN_GOES_HERE")