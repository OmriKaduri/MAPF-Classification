using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Diagnostics;

namespace CPF_experiment
{
    /// <summary>
    /// This class represents a cooperative pathfinding problem instance. This includes:
    /// - The grid in which the agents are located
    /// - An array of initial state for every agent.
    /// </summary>
    public class ProblemInstance
    {
        /// <summary>
        /// Delimiter used for export/import purposes
        /// </summary>
        private static readonly char EXPORT_DELIMITER = ',';

        public static readonly string GRID_NAME_KEY = "Grid Name";
        public static readonly string INSTANCE_NAME_KEY = "Instance Name";

        /// <summary>
        /// This contains extra data of this problem instance (used for special problem instances, e.g. subproblems of a bigger problem instance).
        /// </summary>
        public IDictionary<String, Object> parameters;

        public bool[][] m_vGrid;

        /// <summary>
        /// We keep a reference to the array of agents in the original problem.
        /// This will only change when IndependenceDetection's algorithm determines in another
        /// iteration that a new set of agents must be jointly planned due
        /// to their mutual conflicts.
        /// </summary>
        public AgentState[] m_vAgents;
        
        /// <summary>
        /// This is a matrix that contains the cost of the optimal path to the goal of every agent from any point in the grid.
        /// The first dimension of the matrix is the number of agents.
        /// The second dimension of the matrix is the cardinality of the location from which we want the shortest path.
        /// </summary>
        public int[][] singleAgentOptimalCosts;

        /// <summary>
        /// This is a matrix that contains the best move towards the goal of every agent from any point in the grid.
        /// The first dimension of the matrix is the number of agents.
        /// The second dimension of the matrix is the cardinality of the location from which we want the shortest path.
        /// </summary>
        public Move[][] singleAgentOptimalMoves;

        /// <summary>
        /// Matrix that contains for each agent it's distance (shortest path) to goal
        /// </summary>
        public int[] agentDistancesToGoal;

        /// <summary>
        /// Matrix that contains for each two agents the distance between their starting points
        /// </summary>
        public int[,] distanceBetweenAgentStartPoints;

        /// <summary>
        /// Matrix that contains for each two agents the distance between their goals
        /// </summary>
        public int[,] distanceBetweenAgentGoals;

        public uint m_nObstacles;
        public uint m_nLocations;
        public UInt64[] m_vPermutations; // What are these?
        
        /// <summary>
        /// This field is used to identify an instance when running a set of experiments
        /// </summary>
        public int instanceId;
        
        /// <summary>
        /// Enumerates all of the empty spots in the grid. The indices
        /// correspond directly to those used in the grid, where the major
        /// index corresponds to the x-axis and the minor index corresponds to
        /// the y-axis.
        /// </summary>
        public Int32[,] m_vCardinality;

        public ProblemInstance(IDictionary<String,Object> parameters = null)
        {
            if (parameters != null)
                this.parameters = parameters;
            else
                this.parameters = new Dictionary<String, Object>();
        }

        /// <summary>
        /// Create a subproblem of this problem instance, in which only part of the agents are regarded.
        /// </summary>
        /// <param name="selectedAgents">The selected agent states that will be the root of the subproblem.</param>
        /// <returns></returns>
        public ProblemInstance Subproblem(AgentState[] selectedAgents)
        {
            // Notice selected agents may actually be a completely different set of agents.
            // Not copying instance id. This isn't the same problem.
            ProblemInstance subproblemInstance = new ProblemInstance(this.parameters);
            subproblemInstance.Init(selectedAgents, this.m_vGrid, (int)this.m_nObstacles, (int)this.m_nLocations, this.m_vPermutations, this.m_vCardinality);
            subproblemInstance.singleAgentOptimalCosts = this.singleAgentOptimalCosts; // Each subproblem knows every agent's single shortest paths so this.singleAgentOptimalCosts[agent_num] would easily work
            subproblemInstance.singleAgentOptimalMoves = this.singleAgentOptimalMoves;
            return subproblemInstance;
        }

        /// <summary>
        /// Initialize the members of this object, such that the given agent states are the start state of this instance.
        /// </summary>
        /// <param name="agentStartStates"></param>
        /// <param name="grid"></param>
        public void Init(AgentState[] agentStartStates, bool[][] grid, int nObstacles=-1, int nLocations=-1, ulong[] permutations=null, int[,] cardinality=null)
        {
            m_vAgents = agentStartStates;
            m_vGrid = grid;
            
            if (nObstacles == -1)
                m_nObstacles = (uint)grid.Sum(row => row.Count(x => x));
            else
                m_nObstacles = (uint)nObstacles;

            if (nLocations == -1)
                m_nLocations = ((uint)(grid.Length * grid[0].Length)) - m_nObstacles;
            else
                m_nLocations = (uint)nLocations;
            
            if (permutations == null)
                PrecomputePermutations();
            else
                m_vPermutations = permutations;

            if (cardinality == null)
                PrecomputeCardinality();
            else
                m_vCardinality = cardinality;

            agentDistancesToGoal = new int[m_vAgents.Length];
            distanceBetweenAgentGoals = new int[m_vAgents.Length,m_vAgents.Length];
            distanceBetweenAgentStartPoints = new int[m_vAgents.Length,m_vAgents.Length];
        }

        
        /// <summary>
        /// Compute the shortest path to the goal of every agent in the problem instance, from every location in the grid.
        /// Current implementation is a simple breadth-first search from every location in the graph.
        /// </summary>
        public void ComputeSingleAgentShortestPaths()
        {
            Debug.WriteLine("Computing the single agent shortest path for all agents...");
            Console.WriteLine("Computing the single agent shortest path for all agents...");
            //return; // Add for generator

            this.singleAgentOptimalCosts = new int[this.GetNumOfAgents()][];
            this.singleAgentOptimalMoves = new Move[this.GetNumOfAgents()][];

            for (int agentId = 0; agentId < this.GetNumOfAgents(); agentId++)
            {
                // Run a single source shortest path algorithm from the _goal_ of the agent

                // Create initial state
                var agentStartState = this.m_vAgents[agentId];
                var agent = agentStartState.agent;
                var goalState = new AgentState(agent.Goal.x, agent.Goal.y, -1, -1, agentId);

                var result = AllShortestPathsTo(goalState);
                var shortestPathLengths = result.Item1;
                var optimalMoves = result.Item2;

                int start = this.GetCardinality(agentStartState.lastMove);
                if (shortestPathLengths[start] == -1)
                {
                    throw new Exception(String.Format("Unsolvable instance! Agent {0} cannot reach its goal", agentId));
                }

                this.agentDistancesToGoal[agentId] = shortestPathLengths[start];
                this.singleAgentOptimalCosts[agentId] = shortestPathLengths;
                this.singleAgentOptimalMoves[agentId] = optimalMoves;

                for(int otherAgentId=0; otherAgentId< this.GetNumOfAgents(); otherAgentId++)
                {
                    var otherAgentState = this.m_vAgents[otherAgentId];
                    this.distanceBetweenAgentGoals[agentId, otherAgentId] = GetSingleAgentOptimalCost(agentId, otherAgentState.agent.Goal); //Distance from this agent to other agent goal
                    this.distanceBetweenAgentStartPoints[agentId, otherAgentId] = ShortestPathFromAToB(agentStartState, otherAgentState.lastMove);
                }

            }
        }

        /// <summary>
        /// Computes the shortest path to the goal for a given agent from every location in the grid.
        /// Current implementation is a simple breadth-first search from every location in the graph.
        /// </summary>
        /// <param name="state">Agent's goal state</param>
        /// <returns>Tuple with shortestPathLengths and optimalMoves </returns>
        public Tuple<int[],Move[]> AllShortestPathsTo(AgentState state)
        {
            var openlist = new Queue<AgentState>();
            var shortestPathLengths = new int[this.m_nLocations];
            var optimalMoves = new Move[this.m_nLocations];

            for (int i = 0; i < m_nLocations; i++)
                shortestPathLengths[i] = -1;

            openlist.Enqueue(state);
            
            int goalIndex = this.GetCardinality(state.lastMove);
            shortestPathLengths[goalIndex] = 0;
            optimalMoves[goalIndex] = new Move(state.lastMove);
            while (openlist.Count > 0)
            {
                AgentState nextState = openlist.Dequeue();
                // Generate child states
                foreach (TimedMove aMove in nextState.lastMove.GetNextMoves())
                {
                    if (IsValid(aMove))
                    {
                        int entry = m_vCardinality[aMove.x, aMove.y];
                        // If move will generate a new or better state - add it to the queue
                        if ((shortestPathLengths[entry] == -1) || (shortestPathLengths[entry] > nextState.g + 1))
                        {
                            var childState = new AgentState(nextState);
                            childState.MoveTo(aMove);
                            shortestPathLengths[entry] = childState.g;
                            optimalMoves[entry] = new Move(aMove.GetOppositeMove());
                            openlist.Enqueue(childState);
                        }
                    }
                }
            }
            return Tuple.Create<int[], Move[]>(shortestPathLengths,optimalMoves);
        }

        /// <summary>
        /// Compute shortest path from starting state to goal
        /// </summary>
        /// <param name="startState"></param>
        /// <param name="goal"></param>
        /// <returns></returns>
        public int ShortestPathFromAToB(AgentState startState, Move goal)
        {
            var result = AllShortestPathsTo(startState);
            var shortestPathLengths = result.Item1;

            int start = this.GetCardinality(goal);
            if (shortestPathLengths[start] == -1)
            {
                return -1;
            }

            return shortestPathLengths[start];
        }

        /// <summary>
        /// Returns the length of the shortest path between a given coordinate and the goal location of the given agent.
        /// </summary>
        /// <param name="agentNum"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns>The length of the shortest path from x,y to the goal of the agent.</returns>
        public int GetSingleAgentOptimalCost(int agentNum, int x, int y)
        {
            return this.singleAgentOptimalCosts[agentNum][this.m_vCardinality[x, y]];
        }

        /// <summary>
        /// Returns the length of the shortest path between a given coordinate and the goal location of the given agent.
        /// </summary>
        /// <param name="agentNum"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns>The length of the shortest path from x,y to the goal of the agent.</returns>
        public int GetSingleAgentOptimalCost(int agentNum, Move move)
        {
            return this.singleAgentOptimalCosts[agentNum][this.m_vCardinality[move.x, move.y]];
        }

        /// <summary>
        /// Returns the length of the shortest path between a given agent's location and the goal of that agent.
        /// </summary>
        /// <param name="agent"></param>
        /// <returns>The length of the shortest path between a given agent's location and the goal of that agent</returns>
        public int GetSingleAgentOptimalCost(AgentState agentState)
        {
            return this.singleAgentOptimalCosts[agentState.agent.agentNum][this.m_vCardinality[agentState.lastMove.x, agentState.lastMove.y]];
        }

        /// <summary>
        /// Returns the optimal move towards the goal of the given agent. Move isn't necessarily unique.
        /// </summary>
        /// <param name="agent"></param>
        /// <returns></returns>
        public Move GetSingleAgentOptimalMove(AgentState agentState)
        {
            return this.singleAgentOptimalMoves[agentState.agent.agentNum][this.m_vCardinality[agentState.lastMove.x, agentState.lastMove.y]];
        }

        /// <summary>
        /// The returned plan wasn't constructed considering a CAT, so it's possible there's an alternative plan with the same cost and less collisions.
        /// </summary>
        /// <param name="agentState"></param>
        /// <returns></returns>
        public SinglePlan GetSingleAgentOptimalPlan(AgentState agentState,
                                                    out Dictionary<int, int> conflictCountPerAgent, out Dictionary<int, List<int>> conflictTimesPerAgent)
        {
            LinkedList<Move> moves = new LinkedList<Move>();
            int agentNum = agentState.agent.agentNum;
            var conflictCounts = new Dictionary<int, int>();
            var conflictTimes = new Dictionary<int, List<int>>();
            IReadOnlyDictionary<TimedMove, List<int>> CAT;
            if (this.parameters.ContainsKey(CBS_LocalConflicts.CAT)) // TODO: Add support for IndependenceDetection's CAT
                CAT = ((IReadOnlyDictionary<TimedMove, List<int>>)this.parameters[CBS_LocalConflicts.CAT]);
            else
                CAT = new Dictionary<TimedMove, List<int>>();

            TimedMove current = agentState.lastMove; // The starting position
            int time = current.time;

            while (true)
            {
                moves.AddLast(current);

                // Count conflicts:
                current.UpdateConflictCounts(CAT, conflictCounts, conflictTimes);

                if (agentState.agent.Goal.Equals(current))
                    break;

                // Get next optimal move
                time++;
                Move optimal = this.singleAgentOptimalMoves[agentNum][this.GetCardinality(current)];
                current = new TimedMove(optimal, time);
            }

            conflictCountPerAgent = conflictCounts;
            conflictTimesPerAgent = conflictTimes;
            return new SinglePlan(moves, agentNum);
        }

        /// <summary>
        /// Compute Average distance between starting points of all agents
        /// </summary>
        /// <returns></returns>
        public float AverageStartDistances()
        {
            int sumOfStartDistances = 0;
            int counter = 0;
            for(int i = 0; i < distanceBetweenAgentStartPoints.GetLength(0); i++)
            {
                for(int j = i; j < distanceBetweenAgentStartPoints.GetLength(1); j++) //Symmetric matrix therfore iterate only half of it
                {
                    sumOfStartDistances += distanceBetweenAgentStartPoints[i, j];
                    counter++;
                }
            }
            return (float)sumOfStartDistances / (float)counter;
        }

        /// <summary>
        /// Compute Average distance between goals of all agents
        /// </summary>
        /// <returns></returns>
        public float AverageGoalDistances()
        {
            int sumOfGoalDistances = 0;
            int counter = 0;
            for (int i = 0; i < distanceBetweenAgentGoals.GetLength(0); i++)
            {
                for (int j = i; j < distanceBetweenAgentGoals.GetLength(1); j++) //Symmetric matrix therfore iterate only half of it
                {
                    sumOfGoalDistances += distanceBetweenAgentGoals[i, j];
                    counter++;
                }
            }
            return (float)sumOfGoalDistances / (float)counter;
        }

        /// <summary>
        /// Compute the ratio of points at the grid which are part of a shortest path
        /// </summary>
        /// <returns></returns>
        public float RatioOfPointsAtSP()
        {
            bool[,] pointAtSP = new bool[m_vGrid.GetLength(0), m_vGrid[0].GetLength(0)];
            int NumOfPointsAtSp = 0;
            for (int agentId = 0; agentId < this.GetNumOfAgents(); agentId++)
            {
                var agentStartState = this.m_vAgents[agentId];
                var conflictCountsPerAgent = new Dictionary<int, int>[this.GetNumOfAgents()]; 
                var conflictTimesPerAgent = new Dictionary<int, List<int>>[this.GetNumOfAgents()];
                var optimalPlan = GetSingleAgentOptimalPlan(agentStartState, out conflictCountsPerAgent[agentId], out conflictTimesPerAgent[agentId]);
                foreach(Move currMove in optimalPlan.locationAtTimes){
                    pointAtSP[currMove.x, currMove.y] = true;
                }
            }
            
            for(int row = 0; row < pointAtSP.GetLength(0); row++)
            {
                for(int col=0;col < pointAtSP.GetLength(1); col++)
                {
                    NumOfPointsAtSp += pointAtSP[row,col] ? 1 : 0;
                }
            }

            return (float)NumOfPointsAtSp / ((float)pointAtSP.Length);
        }

        /// <summary>
        /// Utility function that returns the number of agents in this problem instance.
        /// </summary>
        public int GetNumOfAgents()
        {
            return m_vAgents.Length;
        }

        /// <summary>
        /// Utility function that returns the x dimension of the grid
        /// </summary>
        public int GetMaxX()
        {
            return this.m_vGrid.GetLength(0);
        }

        /// <summary>
        /// Utility function that returns the y dimension of the grid
        /// </summary>
        public int GetMaxY()
        {
            return this.m_vGrid[0].Length;
        }

        /// <summary>
        /// Roni: I am not sure when should this be used. It doesn't initialize the grid, 
        /// so I assume that this is meant to be used when a single problem instance object is used and 
        /// modified during the search. This should be used with caution, as we are talking about references
        /// (so if one will change m_vAgents, all the other references to that instance will also point to the same, changed, instance.
        /// </summary>
        /// <param name="ags"></param>
        [Obsolete("Need to have some justification for using this. Currently I believe it will always cause bugs.")]
        public void Init(AgentState[] ags)
        {
            m_vAgents = ags;
            PrecomputePermutations();
        }
        
        public static ProblemInstance ImportFromAgentsFile(String fileName, string mapFilePath = null)
        {
            string fileNameWithoutExtension = Path.GetFileNameWithoutExtension(fileName);
            int instanceId = 0;
            string mapfileName;
            if (mapFilePath == null)
            {
                mapfileName = fileName.Substring(0, fileName.IndexOf('_') + 1 + 1) + ".map"; // FIXME: only supports singl
                mapFilePath = Path.Combine(Path.GetDirectoryName(fileName), "..", "maps", mapfileName);
                instanceId = int.Parse(fileName.Split('_').Last());
            }
            else
            {
                mapfileName = Path.GetFileNameWithoutExtension(mapFilePath);
            }

            Console.WriteLine("map file name: {0} ", mapfileName);

            bool[][] grid;
            string line;
            string[] lineParts;
            using (TextReader input = new StreamReader(mapFilePath))
            {
                // Read grid dimensions
                line = input.ReadLine();
                lineParts = line.Split(',');
                int maxX = int.Parse(lineParts[0]);
                int maxY = int.Parse(lineParts[1]);
                grid = new bool[maxX][];
                char cell;
                // Read grid
                for (int i = 0; i < maxX; i++)
                {
                    grid[i] = new bool[maxY];
                    line = input.ReadLine();
                    for (int j = 0; j < maxY; j++)
                    {
                        cell = line.ElementAt(j);
                        if (cell == '1')
                            grid[i][j] = true;
                        else
                            grid[i][j] = false;
                    }
                }
            }

            AgentState[] states;
            using (TextReader input = new StreamReader(fileName))
            {
                // Read the number of agents
                line = input.ReadLine();
                int numOfAgents = int.Parse(line);

                // Read the agents' start and goal states
                states = new AgentState[numOfAgents];
                AgentState state;
                Agent agent;
                int agentNum;
                int goalX;
                int goalY;
                int startX;
                int startY;
                for (int i = 0; i < numOfAgents; i++)
                {
                    line = input.ReadLine();
                    lineParts = line.Split(EXPORT_DELIMITER);
                    //agentNum = int.Parse(lineParts[0]);
                    goalX = int.Parse(lineParts[0]);
                    goalY = int.Parse(lineParts[1]);
                    startX = int.Parse(lineParts[2]);
                    startY = int.Parse(lineParts[3]);
                    agent = new Agent(goalX, goalY, i);
                    state = new AgentState(startX, startY, agent);
                    states[i] = state;
                }
            }

            // Generate the problem instance
            ProblemInstance instance = new ProblemInstance();
            instance.Init(states, grid);
            instance.instanceId = instanceId;
            instance.parameters[ProblemInstance.GRID_NAME_KEY] = mapfileName;
            instance.parameters[ProblemInstance.INSTANCE_NAME_KEY] = fileNameWithoutExtension + ".agents";
            instance.ComputeSingleAgentShortestPaths();
            return instance;
        }

        public static ProblemInstance ImportFromScenFile(string fileName)
        {
            string fileNameWithoutExtension = Path.GetFileNameWithoutExtension(fileName);
            int instanceId = int.Parse(fileNameWithoutExtension.Split('-').Last());
            string mapfileName = fileNameWithoutExtension.Substring(0, length: fileNameWithoutExtension.LastIndexOf("-even"));  // Passing a length parameter is like specifying a non-inclusive end index
            string mapFilePath = Path.Combine(Path.GetDirectoryName(fileName), "..","..", "maps", mapfileName+".map");
            Console.WriteLine("map file path {0} {1}", Path.GetDirectoryName(fileName), mapFilePath);
            bool[][] grid;
            string line;
            string[] lineParts;
            int maxX;
            int maxY;
            using (TextReader input = new StreamReader(mapFilePath))
            {
                // Read grid dimensions
                line = input.ReadLine();
                Debug.Assert(line.StartsWith("type octile"));
                line = input.ReadLine();
                lineParts = line.Split(' ');
                Debug.Assert(lineParts.Length == 2);
                Debug.Assert(lineParts[0].Equals("height"));
                maxY = int.Parse(lineParts[1]);  // The height is the number of rows
                line = input.ReadLine();
                lineParts = line.Split(' ');
                Debug.Assert(lineParts.Length == 2);
                Debug.Assert(lineParts[0].Equals("width"));
                maxX = int.Parse(lineParts[1]);  // The width is the number of columns
                grid = new bool[maxY][];

                line = input.ReadLine();
                Debug.Assert(line.StartsWith("map"));

                char cell;
                // Read grid
                for (int i = 0; i < maxY; i++)
                {
                    grid[i] = new bool[maxX];
                    line = input.ReadLine();
                    for (int j = 0; j < maxX; j++)
                    {
                        cell = line.ElementAt(j);
                        if (cell == '@' || cell == 'O' || cell == 'T' || cell == 'W' /* Water isn't traversable from land */)
                            grid[i][j] = true;
                        else
                            grid[i][j] = false;
                    }
                }
            }

            List<AgentState> stateList = new List<AgentState>();
            Run runner = new Run();
            Console.WriteLine("Starting scen file {0}", fileName);
            using (TextReader input = new StreamReader(fileName))
            {
                // Read the format version number
                line = input.ReadLine();
                lineParts = line.Split(' ');
                Debug.Assert(lineParts[0].Equals("version"));
                int version = int.Parse(lineParts[1]);
                Debug.Assert(version == 1, "Only version 1 is currently supported");

                // Read the agents' start and goal states
                AgentState state;
                Agent agent;
                int agentNum = 0;
                int block;
                int goalX;
                int goalY;
                int startX;
                int startY;
                string mapFileName;
                int mapRows;
                int mapCols;
                double optimalCost;  // Assuming diagonal moves are allowed and cost sqrt(2)
                while (true)
                {
                    line = input.ReadLine();
                    if (string.IsNullOrWhiteSpace(line))
                        break;
                    lineParts = line.Split('\t');
                    block = int.Parse(lineParts[0]);
                    mapFileName = lineParts[1];
                    mapRows = int.Parse(lineParts[2]);
                    Debug.Assert(mapRows == maxX);
                    mapCols = int.Parse(lineParts[3]);
                    Debug.Assert(mapRows == maxY);

                    startY = int.Parse(lineParts[4]);
                    startX = int.Parse(lineParts[5]);
                    goalY = int.Parse(lineParts[6]);
                    goalX = int.Parse(lineParts[7]);
                    optimalCost = double.Parse(lineParts[8]);
                    agent = new Agent(goalX, goalY, agentNum);
                    state = new AgentState(startX, startY, agent);
                    stateList.Add(state);
                    agentNum++;
                    String instanceName;
                    bool resultsFileExisted = File.Exists(Program.RESULTS_FILE_NAME);
                    runner.OpenResultsFile(Program.RESULTS_FILE_NAME);

                    if (resultsFileExisted == false)
                        runner.PrintResultsFileHeader();
                    runner.CloseResultsFile();
                    TextWriter output;

                    string[] cur_lineParts = null;

                    Console.WriteLine("Starting scen with {0} agents", agentNum);
                    // Generate the problem instance
                    ProblemInstance instance = new ProblemInstance();
                    instance.Init(stateList.ToArray(), grid);
                    instance.instanceId = instanceId;
                    instance.parameters[ProblemInstance.GRID_NAME_KEY] = mapfileName;
                    instance.parameters[ProblemInstance.INSTANCE_NAME_KEY] = fileNameWithoutExtension + ".scen";
                    instance.ComputeSingleAgentShortestPaths();
                    runner.OpenResultsFile(Program.RESULTS_FILE_NAME);
                    if (resultsFileExisted == false)
                        runner.PrintResultsFileHeader();
                    Boolean solved = runner.SolveGivenProblem(instance);
                    runner.CloseResultsFile();
                    if (!solved)
                    {
                        break;
                    }
                }
            }

            
            return null;
        }

        /// <summary>
        /// Imports a problem instance from a given file
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public static ProblemInstance Import(string fileName,string mapFilePath = null)
        {
            if (fileName.EndsWith(".agents"))
            {
                return ImportFromAgentsFile(fileName, mapFilePath);
            }
            else if (fileName.EndsWith(".scen"))
            {
                return ImportFromScenFile(fileName);
            }
            else
            {
                TextReader input = new StreamReader(fileName);
                //return new ProblemInstance(); // DELETE ME!!!
                string[] lineParts;
                string line;
                int instanceId = 0;
                string gridName = "Random Grid"; // The default

                line = input.ReadLine();
                if (line.StartsWith("Grid:") == false)
                {
                    lineParts = line.Split(',');
                    instanceId = int.Parse(lineParts[0]);
                    if (lineParts.Length > 1)
                        gridName = lineParts[1];
                    line = input.ReadLine();
                }
                //instanceId = int.Parse(fileName.Split('-')[4]);
                // First/second line is Grid:
                Debug.Assert(line.StartsWith("Grid:"));

                // Read grid dimensions
                line = input.ReadLine();
                lineParts = line.Split(',');
                int maxX = int.Parse(lineParts[0]);
                int maxY = int.Parse(lineParts[1]);
                bool[][] grid = new bool[maxX][];
                char cell;
                for (int i = 0; i < maxX; i++)
                {
                    grid[i] = new bool[maxY];
                    line = input.ReadLine();
                    for (int j = 0; j < maxY; j++)
                    {
                        cell = line.ElementAt(j);
                        if (cell == '@' || cell == 'O' || cell == 'T' || cell == 'W' /* Water isn't traversable from land */)
                            grid[i][j] = true;
                        else
                            grid[i][j] = false;
                    }
                }

                // Next line is Agents:
                line = input.ReadLine();
                Debug.Assert(line.StartsWith("Agents:"));

                // Read the number of agents
                line = input.ReadLine();
                int numOfAgents = int.Parse(line);

                // Read the agents' start and goal states
                AgentState[] states = new AgentState[numOfAgents];
                AgentState state;
                Agent agent;
                int agentNum;
                int goalX;
                int goalY;
                int startX;
                int startY;
                for (int i = 0; i < numOfAgents; i++)
                {
                    line = input.ReadLine();
                    lineParts = line.Split(EXPORT_DELIMITER);
                    agentNum = int.Parse(lineParts[0]);
                    goalX = int.Parse(lineParts[1]);
                    goalY = int.Parse(lineParts[2]);
                    startX = int.Parse(lineParts[3]);
                    startY = int.Parse(lineParts[4]);
                    agent = new Agent(goalX, goalY, agentNum);
                    state = new AgentState(startX, startY, agent);
                    states[i] = state;
                }

                // Generate the problem instance
                ProblemInstance instance = new ProblemInstance();
                instance.Init(states, grid);
                instance.instanceId = instanceId;
                instance.parameters[ProblemInstance.GRID_NAME_KEY] = gridName;
                instance.parameters[ProblemInstance.INSTANCE_NAME_KEY] = Path.GetFileName(fileName);
                instance.ComputeSingleAgentShortestPaths();
                return instance;
            }
        }

        /// <summary>
        /// Exports a problem instance to a file
        /// </summary>
        /// <param name="fileName"></param>
        public void Export(string fileName)
        {
            TextWriter output = new StreamWriter(Directory.GetCurrentDirectory() + "\\Instances\\"+fileName);
            // Output the instance ID
            if (this.parameters.ContainsKey(ProblemInstance.GRID_NAME_KEY))
                output.WriteLine(this.instanceId.ToString() + "," + this.parameters[ProblemInstance.GRID_NAME_KEY]);
            else
                output.WriteLine(this.instanceId);

            // Output the grid
            output.WriteLine("Grid:");
            output.WriteLine(this.m_vGrid.GetLength(0) + "," + this.m_vGrid[0].GetLength(0));
                        
            for (int i = 0; i < this.m_vGrid.GetLength(0); i++)
            {
                for (int j = 0; j < this.m_vGrid[0].GetLength(0); j++)
                {
                    if (this.m_vGrid[i][j] == true)
                        output.Write('@');
                    else
                        output.Write('.');
                    
                }
                output.WriteLine();
            }
            // Output the agents state
            output.WriteLine("Agents:");
            output.WriteLine(this.m_vAgents.Length);
            AgentState state;
            for(int i = 0 ; i < this.m_vAgents.Length ; i++)
            {
                state = this.m_vAgents[i];
                output.Write(state.agent.agentNum);
                output.Write(EXPORT_DELIMITER);
                output.Write(state.agent.Goal.x);
                output.Write(EXPORT_DELIMITER);
                output.Write(state.agent.Goal.y);
                output.Write(EXPORT_DELIMITER);
                output.Write(state.lastMove.x);
                output.Write(EXPORT_DELIMITER);
                output.Write(state.lastMove.y);
                output.WriteLine();
            }
            output.Flush();
            output.Close();
        }

        /// <summary>
        /// Given an agent located at the nth location on our board that is
        /// not occupied by an obstacle, we return n.
        /// </summary>
        /// <param name="location">An agent's current location.</param>
        /// <returns>n, where the agent is located at the nth non-obstacle
        /// location in our grid.</returns>
        public Int32 GetCardinality(Move location)
        {
            return m_vCardinality[location.x, location.y];
        }
        
        private void PrecomputePermutations()
        {
            m_vPermutations = new UInt64[m_vAgents.Length];
            m_vPermutations[m_vPermutations.Length - 1] = 1;
            for (int i = m_vPermutations.Length - 2; i >= 0; --i)
                m_vPermutations[i] = m_vPermutations[i + 1] * ((UInt64)(m_nLocations - (i + 1)));
            UInt64 m_nPermutations = 1;
            uint nCurrentCounter = m_nLocations;
            for (uint i = 0; i < m_vAgents.Length; ++i)
            {
                m_nPermutations *= nCurrentCounter;
                --nCurrentCounter;
            }
            ++m_nPermutations;
        }
        
        private void PrecomputeCardinality()
        {
            m_vCardinality = new Int32[m_vGrid.Length, m_vGrid[0].Length];
            Int32 maxCardinality = 0;
            for (uint i = 0; i < m_vGrid.Length; ++i)
                for (uint j = 0; j < m_vGrid[i].Length; ++j)
                {
                    if (m_vGrid[i][j])
                        m_vCardinality[i, j] = -1;
                    else
                        m_vCardinality[i, j] = maxCardinality++;
                }
        }

        /// <summary>
        /// Check if the tile is valid, i.e. in the grid and without an obstacle.
        /// NOT checking the direction. A Move could be declared valid even if it came to an edge tile from outside the grid!
        /// NOT checking if the move is illegal
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns>True if the given location is a valid grid location with no obstacles</returns>
        public bool IsValid(Move aMove)
        {
            return !HasObstacleAt(aMove.x, aMove.y);
        }
            
        /// <summary>
        /// Also checks if the move is illegal
        /// </summary>
        /// <param name="toCheck"></param>
        /// <returns></returns>
        public bool IsValid(TimedMove toCheck)
        {
            if (HasObstacleAt(toCheck.x, toCheck.y))
                return false;

            if (parameters.ContainsKey(IndependenceDetection.ILLEGAL_MOVES_KEY))
            {
                var reserved = (HashSet<TimedMove>)parameters[IndependenceDetection.ILLEGAL_MOVES_KEY];

                return (toCheck.IsColliding(reserved) == false);
            } // FIXME: Should this be here?

            return true;
        }

        public bool HasObstacleAt(int x, int y)
        {
            if (x < 0 || x >= GetMaxX())
                return true;
            if (y < 0 || y >= GetMaxY())
                return true;
            return m_vGrid[x][y];
        }

        public override string ToString()
        {
            string str = "Problem instance:" + instanceId;
            if (this.parameters.ContainsKey(ProblemInstance.GRID_NAME_KEY))
                str += " Grid Name:" + this.parameters[ProblemInstance.GRID_NAME_KEY];
            str += " #Agents:" + m_vAgents.Length + ", GridCells:" + m_nLocations + ", #Obstacles:" + m_nObstacles;
            return str;
        }
    }
}
