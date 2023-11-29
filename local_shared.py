import os

# state that is shared, available to all modules

ETK_PATH = os.path.dirname(os.path.abspath(__file__))
# this is two levels down
GDE_PATH = os.path.normpath(os.path.join(ETK_PATH, "../../"))
