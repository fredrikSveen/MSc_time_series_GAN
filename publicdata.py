# First time you use it, you need to
#!pip install cognite-sdk

from cognite.client import CogniteClient, ClientConfig
from cognite.client.credentials import OAuthInteractive

# This value will depend on the cluster your CDF project runs on
base_url = 'https://api.cognitedata.com'
tenant_id = '48d5043c-cf70-4c49-881c-c638f5796997'

creds = OAuthInteractive(
    authority_url=f"https://login.microsoftonline.com/{tenant_id}",
    client_id="1b90ede3-271e-401b-81a0-a4d52bea3273",
    scopes=[f"{base_url}/.default"]
    )  

cnf = ClientConfig(
    client_name='publicdata_user',
    project='publicdata',
    credentials=creds,
    base_url=base_url
    ) 
client = CogniteClient(cnf)
