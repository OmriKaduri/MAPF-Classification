## This is a fork of [roni stern's MAPF Project](https://github.com/ronistern/mapf) ##

# MAPF - Multi Agent Path Finding

A suite of multi-agent path finding algorithm implementations, evaluation and algorithm selection models.

## Benchmark evaluations & algorithm selection ##
We conducted several experiments to evaluate the different MAPF algorithms. 
Furthermore, we trained algorithm selection models, based on XGBoost and VGG-16.
The relevant code and data is under `classification` directory, 
with the corresponding documentation to reproduce our results.  

## How to run (solve MAPF problems) ##
In `Run.cs` you need to add your `ISolver` implementation to `solvers` List. 
After your addition, go to `Program.cs`. 
At the main function you can choose to run in either one of the next modes:
1. `runGrids` - You programtically define, using int arrays, the grid size, obstacles and agents. Then you run `RunExperimentSet` function with the given inputs.
2. `runSpecific`  - Will use grid definition file from `bin\Debug\instances`. Currently uses `Instance-4-0-3-0` -  a grid of 4x4 with 3 agents and no obstacles
3. `runNathan` - Will run [Nathan Sturtevant benchmarks](https://movingai.com/benchmarks/mapf/index.html).


## Currently implemented Solvers ##
1. A* and it's variations (Operator Decompsition, PartialExpansion, Independence Detection)
2. CBS and it's variations
3. ICTS
4. [MDD-SAT](https://github.com/surynek/boOX) integration as an `ISolver`.
5. [LazyCBS](https://bitbucket.org/gkgange/lazycbs/src/master/) integration as an `ISolver`. 
