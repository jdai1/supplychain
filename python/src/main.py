from argparse import ArgumentParser
from pathlib import Path
from model_timer import Timer 
from lpinstance import LPSolver, IPSolver
import json


# Stencil created by Anirudh Narsipur March 2023


def main(args):

    filename = Path(args.input_file).name
    timer = Timer()
    timer.start() 
    lpsolver = IPSolver(args.input_file)
    try:
        sol = lpsolver.solve()
        timer.stop()
    except Exception as _:
        printSol = {
            "Instance" : filename,
            "Time" : timer.getElapsed(),
            "Result" : "--",
            "Solution" : "--"
        }
    else:
        printSol = {
            "Instance" : filename,
            "Time" : timer.getElapsed(),
            "Result" : sol, 
            "Solution" : "OPT"
        }
        print(json.dumps(printSol))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()
    main(args)
