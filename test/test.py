import unittest
import os
import numpy as np
import json
import sys
from typing import Dict, List, Tuple

class SupplyChainTest(unittest.TestCase):
    def __init__(self, methodName='runTest', instance_file=None, output_file=None, line_number=0):
        super().__init__(methodName)
        self.instance_file = instance_file
        self.output_file = output_file
        self.line_number = line_number
    
    def setUp(self):
        # Default setup values if not provided in constructor
        if not self.instance_file:
            raise ValueError("Instance file is required")
        if not self.output_file:
            raise ValueError("Output file is required")
       
        # load the instance data
        with open(self.instance_file, "r") as f:
            self.numCustomers, self.numFacilities = [int(i) for i in f.readline().split()]
            allocCostraw = [float(i) for i in f.readline().split()]
            self.allocCostCF = np.zeros((self.numCustomers, self.numFacilities))
            index = 0
            for i in range(self.numCustomers):
                for j in range(self.numFacilities):
                    self.allocCostCF[i, j] = allocCostraw[index]
                    index += 1
            self.demandC = np.array([float(i) for i in f.readline().split()])
            self.openingCostF = np.array([float(i) for i in f.readline().split()])
            self.capacityF = np.array([float(i) for i in f.readline().split()])
            self.truckDistLimit, self.truckUsageCost = [float(i) for i in f.readline().split()]
            distanceCFraw = [float(i) for i in f.readline().split()]
            self.distanceCF = np.zeros((self.numCustomers, self.numFacilities))
            index = 0
            for i in range(self.numCustomers):
                for j in range(self.numFacilities):
                    self.distanceCF[i, j] = distanceCFraw[index]
                    index += 1
            
        # Load the solution output
        self.min_cost, self.matrix_vars, self.vehicle_vars = self.parse_solution(self.output_file, self.line_number)
        print(f"Instance: {self.instance_file}, Output: {self.output_file}")
        print(f"Cost: {self.min_cost}, Matrix shape: {self.matrix_vars.shape}, Vehicle shape: {self.vehicle_vars.shape}")
            
    def parse_solution(self, output_file: str, line_number: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Parse the solution output file into min cost, matrix variables, and vehicle variables
        
        Args:
            output_file: Path to the solution output file
            
        Returns:
            Tuple containing:
            - min_cost: Minimum cost value
            - matrix_vars: Matrix variables as numpy array with shape (numCustomers, numFacilities)
            - vehicle_vars: Vehicle variables as numpy array with length numFacilities
        """
        try:
            with open(output_file, 'r') as f:
                line = f.readlines()[line_number]
                # Parse the JSON content
                data = json.loads(line)
                result_str = data["Result"]
                
            # Extract min cost
            min_cost_str = result_str.split("Minimum Cost: ")[1].split("=====")[0].strip()
            min_cost = float(min_cost_str)
            
            # Extract matrix variables
            matrix_section = result_str.split("===== MATRIX VARIABLES =====")[1].split("===== VEHICLE VARIABLES =====")[0]
            matrix_rows = matrix_section.split('\n')
            
            matrix_values = []
            for row in matrix_rows:
                if row.strip():
                    row_values = [float(val) for val in row.strip().split()]
                    matrix_values.append(row_values)
             
            matrix_vars = np.array(matrix_values)   
            assert matrix_vars.shape == (self.numFacilities, self.numCustomers)

            # Extract vehicle variables
            vehicle_section = result_str.split("===== VEHICLE VARIABLES =====")[1]
            vehicle_vars = []
            for v in vehicle_section.strip().split():
                if v.strip():
                    vehicle_vars.append(float(v))
            assert len(vehicle_vars) == self.numFacilities
            return min_cost, matrix_vars, np.array(vehicle_vars)
            
        except Exception as e:
            print(f"Error parsing solution file: {e}")
            return 0.0, np.array([]), np.array([])
            
    def test_facility_capacity_constraint(self):
        """
        Test that each facility does not exceed its maximum capacity.
        Capacity constraint: Sum of (customer demand * assignment) for each facility <= facility capacity
        """
        for j in range(self.numFacilities):
            facility_load = 0
            for i in range(self.numCustomers):
                facility_load += self.demandC[i] * self.matrix_vars[j, i]
            self.assertLessEqual(facility_load, self.capacityF[j] + 1e-2,
                                f"Facility {j} exceeds capacity: {facility_load} > {self.capacityF[j]}")
    
    def test_customer_assignment_constraint(self):
        """
        Test that each customer is assigned to exactly one facility.
        Assignment constraint: Sum of assignments for each customer = 1
        """
        for i in range(self.numCustomers):
            total_assignment = 0
            for j in range(self.numFacilities):
                total_assignment += self.matrix_vars[j, i]
            self.assertAlmostEqual(total_assignment, 1.0, 
                                  msg=f"Customer {i} assignment sum is {total_assignment}, should be 1.0",
                                  delta=1e-2)
    
    def test_vehicle_distance_constraint(self):
        """
        Test that the total distance traveled by vehicles from each facility doesn't exceed the limit.
        Vehicle distance constraint: Each individual vehicle can travel at most truckDistLimit.
        """
        for f in range(self.numFacilities):
            total_distance = 0
            for c in range(self.numCustomers):
                curr_distance = self.distanceCF[c, f] * self.matrix_vars[f, c]
                if curr_distance > self.truckDistLimit:
                    self.fail(f"Instance: {self.instance_file}, Output: {self.output_file}, Line: {self.line_number}, Facility {f} exceeds distance limit: {curr_distance} > {self.truckDistLimit}, distance: {self.distanceCF[c, f]}, assignment: {self.matrix_vars[f, c]}")
                total_distance += curr_distance
                
            self.assertLessEqual(total_distance, self.vehicle_vars[f] * self.truckDistLimit + 1e-2,
                                f"Instance: {self.instance_file}, Output: {self.output_file}, Line: {self.line_number}, Facility {f} exceeds distance limit: {total_distance} > {self.vehicle_vars[f] * self.truckDistLimit}")
    
    def test_overall_solution_validity(self):
        """
        Comprehensive test to ensure all constraints are satisfied and the solution is valid.
        """
        # Check if any facility with assignments has vehicles
        for j in range(self.numFacilities):
            facility_assignments = sum(self.matrix_vars[j, i] for i in range(self.numCustomers))
            if facility_assignments > 0:
                self.assertGreater(self.vehicle_vars[j], 0, 
                                  f"Instance: {self.instance_file}, Output: {self.output_file}, Line: {self.line_number}, Facility {j} has customer assignments but no vehicles")
        
        # Check if solution has reasonable cost (not negative)
        self.assertGreaterEqual(self.min_cost, 0, f"Instance: {self.instance_file}, Output: {self.output_file}, Line: {self.line_number}, Solution cost cannot be negative")
        
        # Verify that matrix variables are within valid bounds [0,1]
        for j in range(self.numFacilities):
            for i in range(self.numCustomers):
                self.assertGreaterEqual(self.matrix_vars[j, i], 0, 
                                      f"Instance: {self.instance_file}, Output: {self.output_file}, Line: {self.line_number}, Assignment variable for facility {j}, customer {i} is negative")
                self.assertLessEqual(self.matrix_vars[j, i], 1 + 1e-6, 
                                    f"Instance: {self.instance_file}, Output: {self.output_file}, Line: {self.line_number}, Assignment variable for facility {j}, customer {i} is greater than 1")
        
        # Verify that vehicle variables are non-negative
        for j in range(self.numFacilities):
            self.assertGreaterEqual(self.vehicle_vars[j], 0, 
                                   f"Instance: {self.instance_file}, Output: {self.output_file}, Line: {self.line_number},  Vehicle count for facility {j} is negative")
            
def create_test_suite(instance_file, output_file, line_number):
    """Create a test suite for a specific instance and output file pair"""
    suite = unittest.TestSuite()
    test_cases = [
        # 'test_facility_capacity_constraint',
        # 'test_customer_assignment_constraint',
        'test_vehicle_distance_constraint',
        # 'test_overall_solution_validity'
    ]
    
    for test_case in test_cases:
        suite.addTest(SupplyChainTest(test_case, instance_file, output_file, line_number))
    
    return suite

def run_tests(result_file):
    master_suite = unittest.TestSuite()
    
    with open(result_file, "r") as f:
        for line_number, line in enumerate(f):
            data = json.loads(line)
            instance_file = "python/input/" + data["Instance"]
            output_file = result_file
            
            print(f"TESTING: {instance_file} Line: {line_number}")
            suite = create_test_suite(instance_file, output_file, line_number)
            master_suite.addTest(suite)
            
    # Run all tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(master_suite)
    
    # Return success/failure
    return result.wasSuccessful()

if __name__ == '__main__': 
    success = run_tests("python/temp.log")

