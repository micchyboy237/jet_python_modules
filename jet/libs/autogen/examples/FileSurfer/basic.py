from autogen_core.models import ChatCompletionClient
from autogen_ext.agents.file_surfer import FileSurfer

# Create a model client (must support tool use)
model_client = ChatCompletionClient(model="gpt-4-tools")

# Initialize FileSurfer
file_surfer = FileSurfer(
    name="LocalFileAgent",
    model_client=model_client,
    base_path="/Users/jethroestrada/Documents"
)
