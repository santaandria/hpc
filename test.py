from dask.distributed import Client
import dask.array as da
import argparse

# Argument parsing to get the scheduler file path
parser = argparse.ArgumentParser(description="Dask Scheduler Connection Example")
parser.add_argument(
    "--scheduler-file", type=str, help="Path to the Dask scheduler file"
)
args = parser.parse_args()

# Connecting to the Dask cluster
client = Client(scheduler_file=args.scheduler_file)

# Example Dask operation
x = da.random.random((10000, 10000), chunks=(1000, 1000))
y = x + x.T
z = y.mean(axis=0)
result = z.compute()

print(result)
print("DONE")
