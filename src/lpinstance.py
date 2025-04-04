from __future__ import annotations
from dataclasses import dataclass

from docplex.mp.context import * # type: ignore
from docplex.mp.model import Model # type: ignore
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
            # print(
            #     f"numCustomers: {numCustomers} numFacilities: {numFacilities} numVehicle: {numMaxVehiclePerFacility}"
            # )
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

            for c in range(self.lpinst.numCustomers):
                self.model.add_constraint(
                    self.matrix_vars[f, c] * self.lpinst.distanceCF[c, f] <= self.lpinst.truckDistLimit
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
            print(f"Minimum Cost: {cost_celing:.2f}")
            
            print("\n===== MATRIX VARIABLES =====")
            for f in range(self.lpinst.numFacilities):
                for c in range(self.lpinst.numCustomers):
                    print(f"{self.matrix_vars[f, c].solution_value:.4f}", end=" ")
                print()
            
            print("\n===== VEHICLE VARIABLES =====")
            for f in range(self.lpinst.numFacilities):
                print(f"Facility {f+1} vehicles: {self.vehicle_vars[f].solution_value:.4f}")
            
            return cost_celing
            
        raise Exception("no solution found")


class IPSolver:
    def __init__(self, filename: str):
        self.lpinst = getLPInstance(filename)
        self.model = Model()  # CPLEX solver
        self.matrix_vars, self.facility_indicators, self.vehicle_indicators = self.init_vars() # vars[c][f][v]

    def init_vars(self):
        vars = []
        assert self.lpinst
        
        for c in range(self.lpinst.numCustomers):
            outer = []
            for f in range(self.lpinst.numFacilities):
                inner = []
                for v in range(self.lpinst.numMaxVehiclePerFacility):
                    inner.append(
                        self.model.integer_var(
                            0, 1, f"Facility: {f + 1}, Customer: {c + 1}, Vehicle: {v + 1}"
                        )
                    )
                outer.append(inner)
            vars.append(outer)

        facility_indicators = self.model.integer_var_list(self.lpinst.numFacilities, 0, 1)
        vehicle_indicators = [self.model.integer_var_list(self.lpinst.numMaxVehiclePerFacility, 0, 1) for _ in range(self.lpinst.numFacilities)] # [f][v]

        return np.array(vars), np.array(facility_indicators), np.array(vehicle_indicators)
    
    def solve(self):
        assert self.lpinst

        for c in range(self.lpinst.numCustomers):
            self.model.add_constraint(self.model.sum(self.matrix_vars[c, :, :].flatten().tolist()) == 1)

        for f in range(self.lpinst.numFacilities):
            for v in range(self.lpinst.numMaxVehiclePerFacility):
                self.model.add_constraint(
                    self.model.scal_prod(
                        terms=self.matrix_vars[:, f, v].flatten().tolist(),
                        coefs=self.lpinst.distanceCF[:, f].tolist(),
                    )
                    <= self.lpinst.truckDistLimit
                )
                for c in range(self.lpinst.numCustomers):
                    self.model.add_constraint(
                        self.vehicle_indicators[f][v] >= self.matrix_vars[c, f, v]
                    )
                self.model.add_constraint(
                    self.vehicle_indicators[f][v] <= self.model.sum(self.matrix_vars[:, f, v].flatten().tolist())
                )

            # terms --> indicator variables for whether factory f serves customers 1..c
            terms = [self.model.sum(self.matrix_vars[c, f, :].flatten().tolist()) for c in range(self.lpinst.numCustomers)]
            self.model.add_constraint(
                self.model.scal_prod(
                    terms=terms,
                    coefs=self.lpinst.demandC.tolist(),
                )
                <= self.lpinst.capacityF[f]
            )

            for c in range(self.lpinst.numCustomers):
                for v in range(self.lpinst.numMaxVehiclePerFacility):
                    self.model.add_constraint(
                        self.facility_indicators[f] >= self.matrix_vars[c, f, v]
                    )
            self.model.add_constraint(
                self.facility_indicators[f] <= self.model.sum(self.matrix_vars[:, f, :].flatten().tolist())
            )

        # TODO: Modify costs

        facility_cost = self.model.scal_prod(
            terms=self.facility_indicators,
            coefs=self.lpinst.openingCostF
        )
        vehicle_cost = self.model.sum(self.vehicle_indicators.flatten().tolist()) * self.lpinst.truckUsageCost
        
        customer_cost = 0
        for f in range(self.lpinst.numFacilities):
            for v in range(self.lpinst.numMaxVehiclePerFacility):
                customer_cost += self.model.scal_prod(
                    terms=self.matrix_vars[:, f, v],
                    coefs=self.lpinst.allocCostCF[:, f],
                )

        self.model.minimize(
            self.model.sum([facility_cost, vehicle_cost, customer_cost])
        )

        sol = self.model.solve()
        cost_celing = self.model.objective_value
        if sol:
            self.model.print_information()
            
            return cost_celing
            
        raise Exception("no solution found")


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
