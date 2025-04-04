from argparse import ArgumentParser
from pathlib import Path
from model_timer import Timer 
from lpinstance import LPSolver, IPSolver
import json


# Stencil created by Anirudh Narsipur March 2023
VERBOSE = False

def main(args):

    filename = Path(args.input_file).name
    timer = Timer()
    timer.start() 
    lpsolver = LPSolver(args.input_file)
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
        if VERBOSE:
            # Format the result as a string directly
            result = "===== SOLUTION SUMMARY ====="
            result += f"Minimum Cost: {sol:.2f}"
            
            # Check if lpinst exists before using its attributes
            if lpsolver.lpinst:
                result += "===== MATRIX VARIABLES ====="
                for f in range(lpsolver.lpinst.numFacilities):
                    matrix_line = ""
                    for c in range(lpsolver.lpinst.numCustomers):
                        matrix_line += f"{lpsolver.matrix_vars[f, c].solution_value:.4f} "
                    result += matrix_line + "\n"
                
                result += "===== VEHICLE VARIABLES ====="
                for f in range(lpsolver.lpinst.numFacilities):
                    result += f"{lpsolver.vehicle_vars[f].solution_value:.4f} "
            
            # Print the summary to the console
            print(result)
            
            # Create and print JSON output
            printSol = {
                "Instance" : filename,
                "Time" : timer.getElapsed(),
                "Result" : result,
                "Solution" : "OPT"
            }
            print(json.dumps(printSol))
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
