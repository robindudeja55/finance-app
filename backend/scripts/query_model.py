import os
import sys
import django
import importlib
from pathlib import Path
#git
# Make sure Python can import config (which lives in /app/config)
BASE_DIR = Path(__file__).resolve().parent.parent  # /app
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/query_model.py <app.Model> <filter_expr> [fields...]")
        print("Example: python scripts/query_model.py market.Instrument symbol=AAPL symbol name")
        sys.exit(1)

    model_path = sys.argv[1]
    filter_expr = sys.argv[2]
    fields = sys.argv[3:]

    try:
        app_label, model_name = model_path.split(".")
        models_module = importlib.import_module(f"{app_label}.models")
        Model = getattr(models_module, model_name)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        sys.exit(1)

    try:
        k, v = filter_expr.split("=", 1)
        qs = Model.objects.filter(**{k: v})
    except Exception as e:
        print(f"Invalid filter '{filter_expr}': use key=value (supports lookups like instrument__symbol=AAPL). Error: {e}")
        sys.exit(1)

    count = qs.count()
    print(f"Found {count} objects for {model_path} where {filter_expr}")

    for obj in qs[:20]:
        if fields:
            parts = [f"{f}={getattr(obj, f, None)}" for f in fields]
            print(", ".join(parts))
        else:
            print(obj)

if __name__ == "__main__":
    main()
