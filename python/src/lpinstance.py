from __future__ import annotations
from dataclasses import dataclass

from docplex.mp.context import *
from docplex.mp.model import Model
from typing import Optional
import numpy as np
import math


@dataclass()
class LPInstance:
    numCustomers: int  # the number of customers
    numFacilities: int  # the number of facilities
    allocCostCF: np.ndarray  # allocCostCF[c][f] is the service cost paid each time customer c is served by facility f
    demandC: np.ndarray  # demandC[c] is the demand of customer c
    openingCostF: np.ndarray  # openingCostF[f] is the opening cost of facility f
    capacityF: np.ndarray  # capacityF[f] is the capacity of facility f
    numMaxVehiclePerFacility: (
        int  # maximum number of vehicles to use at an open facility
    )
    truckDistLimit: float  # total driving distance limit for trucks
    truckUsageCost: float  # fixed usage cost paid if a truck is used
    distanceCF: (
        np.ndarray
    )  # distanceCF[c][f] is the roundtrip distance between customer c and facility f


def getLPInstance(fileName: str) -> Optional[LPInstance]:
    try:
        with open(fileName, "r") as fl:
            numCustomers, numFacilities = [int(i) for i in fl.readline().split()]
            numMaxVehiclePerFacility = numCustomers
            print(
                f"numCustomers: {numCustomers} numFacilities: {numFacilities} numVehicle: {numMaxVehiclePerFacility}"
            )
            allocCostCF = np.zeros((numCustomers, numFacilities))

            allocCostraw = [float(i) for i in fl.readline().split()]
            index = 0
            for i in range(numCustomers):
                for j in range(numFacilities):
                    allocCostCF[i, j] = allocCostraw[index]
                    index += 1

            demandC = np.array([float(i) for i in fl.readline().split()])

            openingCostF = np.array([float(i) for i in fl.readline().split()])

            capacityF = np.array([float(i) for i in fl.readline().split()])

            truckDistLimit, truckUsageCost = [float(i) for i in fl.readline().split()]

            distanceCF = np.zeros((numCustomers, numFacilities))
            distanceCFraw = [float(i) for i in fl.readline().split()]
            index = 0
            for i in range(numCustomers):
                for j in range(numFacilities):
                    distanceCF[i, j] = distanceCFraw[index]
                    index += 1
            return LPInstance(
                numCustomers=numCustomers,
                numFacilities=numFacilities,
                allocCostCF=allocCostCF,
                demandC=demandC,
                openingCostF=openingCostF,
                capacityF=capacityF,
                numMaxVehiclePerFacility=numMaxVehiclePerFacility,
                truckDistLimit=truckDistLimit,
                truckUsageCost=truckUsageCost,
                distanceCF=distanceCF,
            )

    except Exception as e:
        print(f"Could not read problem instance file due to error: {e}")
        return None


class LPSolver:
    def __init__(self, filename: str):
        self.lpinst = getLPInstance(filename)
        self.model = Model()  # CPLEX solver
        self.matrix_vars, self.vehicle_vars = self.init_vars()  # vars[f][c]

    def init_vars(self):
        vars = []
        assert self.lpinst
        for f in range(self.lpinst.numFacilities):
            row = []
            for c in range(self.lpinst.numCustomers):
                row.append(
                    self.model.continuous_var(
                        0, 1, f"Facility: {f + 1}, Customer: {c + 1}"
                    )
                )
            vars.append(row)

        vehicle_vars = []
        for f in range(self.lpinst.numFacilities):
            vehicle_vars.append(
                self.model.continuous_var(
                    0,
                    self.lpinst.numMaxVehiclePerFacility,
                    f"# Vehicles for facility: {f + 1}",
                )
            )

        return np.array(vars), np.array(vehicle_vars)

    def solve(self):
        assert self.lpinst
        for c in range(self.lpinst.numCustomers):
            self.model.add_constraint(self.model.sum(self.matrix_vars[:, c]) == 1)

        for f in range(self.lpinst.numFacilities):
            self.model.add_constraint(
                self.model.scal_prod(
                    terms=self.matrix_vars[f, :].tolist(),
                    coefs=self.lpinst.demandC.tolist(),
                )
                <= self.lpinst.capacityF[f]
            )

            self.model.add_constraint(
                self.model.scal_prod(
                    terms=self.matrix_vars[f, :].tolist(),
                    coefs=self.lpinst.distanceCF[:, f].tolist(),
                )
                <= self.lpinst.truckDistLimit * self.vehicle_vars[f]
            )

        facility_cost = 0
        for f in range(self.lpinst.numFacilities):
            totalDemandMet = self.model.scal_prod(
                terms=self.matrix_vars[f, :].tolist(),
                coefs=self.lpinst.demandC.tolist(),
            )
            facility_cost += self.lpinst.openingCostF[f] * (totalDemandMet / self.lpinst.capacityF[f])
        
        vehicle_cost = self.model.scal_prod(
            terms=self.vehicle_vars,
            coefs=[
                self.lpinst.truckUsageCost for _ in range(self.lpinst.numFacilities)
            ],
        )
        customer_cost = 0
        for f in range(self.lpinst.numFacilities):
            customer_cost += self.model.scal_prod(
                terms=self.matrix_vars[f, :],
                coefs=self.lpinst.allocCostCF[:, f],
            )

        self.model.minimize(
            self.model.sum([facility_cost, vehicle_cost, customer_cost])
        )

        sol = self.model.solve()
        cost_celing = self.model.objective_value
        if sol:
            self.model.print_information()

            # Print matrix values
            print("Matrix values:")
            for f in range(self.lpinst.numFacilities):
                row_values = []
                for c in range(self.lpinst.numCustomers):
                    row_values.append(f"{self.matrix_vars[f, c].solution_value:.4f}")
                print(f"Facility {f+1}: {row_values}")
            
            return cost_celing
            
        raise Exception("balls")


def dietProblem():
    # Diet Problem from Lecture Notes
    m = Model()
    # Note that these are continous variables and not integers
    mvars = m.continuous_var_list(2, 0, 1000)
    carbs = m.scal_prod(terms=mvars, coefs=[100, 250])
    m.add_constraint(carbs >= 500)
    m.add_constraint(m.scal_prod(terms=mvars, coefs=[100, 50]) >= 250)  # Fat
    m.add_constraint(m.scal_prod(terms=mvars, coefs=[150, 200]) >= 600)  # Protein

    m.minimize(m.scal_prod(terms=mvars, coefs=[25, 15]))

    sol = m.solve()
    obj_value = math.ceil(m.objective_value)
    if sol:
        m.print_information()
        print(f"Meat: {mvars[0].solution_value}")
        print(f"Bread: {mvars[1].solution_value}")
        print(f"Objective Value: {obj_value}")
    else:
        print("No solution found!")
