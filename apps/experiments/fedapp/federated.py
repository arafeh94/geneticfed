from src.app.federated_app import FederatedApp
from src.app.settings import Settings

config = Settings.from_json_file('./config.json')
app = FederatedApp(config)
app.start()
