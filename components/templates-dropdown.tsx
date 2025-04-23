"use client"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuGroup,
} from "@/components/ui/dropdown-menu"
import { FileCode, ChevronDown } from "lucide-react"

interface TemplatesDropdownProps {
  setCode: (code: string) => void
}

const TEMPLATES = [
  {
    name: "Tic Tac Toe",
    code: `import random

board = [' '] * 9
game_state = 'PLAYING'

def display_board():
    print(board[0] + " | " + board[1] + " | " + board[2])
    print("---------")
    print(board[3] + " | " + board[4] + " | " + board[5])
    print("---------")
    print(board[6] + " | " + board[7] + " | " + board[8])

def check_win(player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  
        [0, 4, 8], [2, 4, 6] 
    ]
    
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] == player: # player = 'O' or 'X'
            return True
    return False


def check_draw():
    return ' ' not in board

def available_moves():
    moves = []
    for i in range(len(board)):
        if board[i] == ' ':
            moves.append(i)
    return moves

def player_turn():
    while True:
        try:
            move = int(input("Enter your move (1-9): ")) - 1
            if move in available_moves():
                board[move] = 'X'
                break
            else:
                print("Invalid move. Try again.")
        except ValueError:
            print("Please enter a valid number between 1 and 9.")


def computer_turn():
    for move in available_moves(): # if computer is winning
        board[move] = 'O'
        if check_win('O'):
            return
        board[move] = ' '

    for move in available_moves(): # if player is winning block them
        board[move] = 'X'
        if check_win('X'):
            board[move] = 'O'
            return
        board[move] = ' '

    move = random.choice(available_moves()) # otherwise random value
    board[move] = 'O'

while game_state == 'PLAYING':
    display_board()

    player_turn()
    if check_win('X'):
        game_state = 'USER_WINS'
        break
    if check_draw():
        game_state = 'DRAW'
        break

    computer_turn()
    if check_win('O'):
        game_state = 'COMPUTER_WINS'
        break
    if check_draw():
        game_state = 'DRAW'
        break

display_board()
if game_state == 'USER_WINS':
    print("You win!")
elif game_state == 'COMPUTER_WINS':
    print("Computer wins!")
else:
    print("It's a draw!")
`,
  },
  {
    name: "waterjug",
    code: `def heuristic(x, y, Z):  # heuristic function for calculation of best path
    return abs(Z - x) + abs(Z - y)

def get_next_states(x, y, X, Y):
    return [
        (X, y),  # fill X
        (x, Y),  # fill Y
        (0, y),  # empty X
        (x, 0),  # empty Y
        (x - min(x, Y - y), y + min(x, Y - y)), # x -> y
        (x + min(y, X - x), y - min(y, X - x))  # y -> x
    ]

def hill_climbing(X, Y, Z):
    current = (0, 0)
    visited = set()
    path = [current]

    while True:
        if Z in current:
            print("Solution found:", path)
            return

        visited.add(current)
        neighbors = get_next_states(*current, X, Y)

        best_state = None
        best_h = 1e8

        for state in neighbors:
            if state not in visited:
                h = heuristic(*state, Z)
                # print(state)
                if h < best_h:
                    best_h = h
                    best_state = state

        if best_state is None:
            print("Stuck! No solution found.")
            return

        current = best_state
        path.append(current)

# x=4, y=3, target=2
hill_climbing(4, 3, 2)
`,
  },
  {
    name: "8puzzle_hillclimb",
    code: `import java.util.*;

class EightPuzzleHillClimbing {
    static int[][] goal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 0 } };
    static int[][] moves = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };

    static int heuristic(int[][] state) {
        int h = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (state[i][j] != 0) {
                    int val = state[i][j] - 1; // indexing to 0
                    int goalX = val / 3, goalY = val % 3; // manhattan distance heuristic for each tile to its goal position
                    h += Math.abs(i - goalX) + Math.abs(j - goalY); // |x - goalX| + |y - goalY|
                }
            }
        }
        return h;
    }

    static int[][] getBestNeighbor(int[][] state) {
        int bestH = heuristic(state);
        int[][] bestState = state;

        for (int[] move : moves) {
            int[][] newState = moveTile(state, move);
            if (newState != null) {
                int newH = heuristic(newState);
                if (newH < bestH) {
                    bestH = newH;
                    bestState = newState;
                }
            }
        }
        return bestState;
    }

    static int[][] moveTile(int[][] state, int[] move) {
        int x = 0, y = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (state[i][j] == 0) { // finding the empty tile
                    x = i;
                    y = j;
                } 

        int nx = x + move[0], ny = y + move[1];
        if (nx >= 0 && ny >= 0 && nx < 3 && ny < 3) {
            int[][] newState = new int[3][3];
            for (int i = 0; i < 3; i++)
                newState[i] = state[i].clone();
            newState[x][y] = newState[nx][ny];
            newState[nx][ny] = 0;
            return newState;
        }
        return null;
    }

    static void solve(int[][] start) {
        int[][] current = start;
        while (!Arrays.deepEquals(current, goal)) {
            System.out.println("Before");
            printState(current);
            int[][] next = getBestNeighbor(current);
            System.out.println("After");
            printState(next);
            if (Arrays.deepEquals(next, current)) {
                System.out.println("Stuck in local optimum! No solution found.");
                return;
            }
            current = next;
            // printState(current);
        }
        System.out.println("Solution found!");
    }

    static void printState(int[][] state) {
        for (int[] row : state)
            System.out.println(Arrays.toString(row));
        System.out.println();
    }

    public static void main(String[] args) {
        int[][] start = { { 1, 2, 3 }, { 4, 0, 6 }, { 7, 5, 8 } };
        solve(start);
    }
}
`,
  },
  {
    name: "8puzzle_gbfs_astar",
    code: `import java.util.*;

class EightPuzzleGBFS {
    static int[][] goal = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 0}
    };

    static int heuristic(int[][] state) {
        int h = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (state[i][j] != 0 && state[i][j] != goal[i][j]) h++;
            }
        }
        return h;
    }

    static List<int[][]> getNeighbors(int[][] state) {
        List<int[][]> neighbors = new ArrayList<>();
        int x = 0, y = 0;
        
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                if (state[i][j] == 0) { x = i; y = j; }

        int[][] moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (int[] move : moves) {
            int nx = x + move[0], ny = y + move[1];
            if (nx >= 0 && ny >= 0 && nx < 3 && ny < 3) {
                int[][] newState = new int[3][3];
                for (int i = 0; i < 3; i++) newState[i] = state[i].clone();
                newState[x][y] = newState[nx][ny];
                newState[nx][ny] = 0;
                neighbors.add(newState);
            }
        }
        return neighbors;
    }

    static void greedyBestFirstSearch(int[][] start) {
        PriorityQueue<int[][]> pq = new PriorityQueue<>(Comparator.comparingInt(EightPuzzleGBFS::heuristic));
        Set<String> visited = new HashSet<>();
        pq.add(start);

        while (!pq.isEmpty()) {
            int[][] current = pq.poll();
            printState(current);
            if (Arrays.deepEquals(current, goal)) {
                System.out.println("Solved!");
                return;
            }

            visited.add(Arrays.deepToString(current));
            for (int[][] neighbor : getNeighbors(current)) {
                if (!visited.contains(Arrays.deepToString(neighbor))) pq.add(neighbor);
            }
        }
        System.out.println("No solution found.");
    }

    static void printState(int[][] state) {
        for (int[] row : state) System.out.println(Arrays.toString(row));
        System.out.println();
    }

    public static void main(String[] args) {
        int[][] start = {
            {1, 2, 3},
            {4, 0, 5},
            {7, 8, 6}
        };
        greedyBestFirstSearch(start);
    }
}


import java.util.*;

class PuzzleNode implements Comparable<PuzzleNode> {
    int[][] state;
    int g, h;
    PuzzleNode parent;
    
    public PuzzleNode(int[][] state, int g, int h, PuzzleNode parent) {
        this.state = state;
        this.g = g;
        this.h = h;
        this.parent = parent;
    }
    
    public int f() {
        return g + h;
    }
    
    @Override
    public int compareTo(PuzzleNode other) {
        return Integer.compare(this.f(), other.f());
    }
}

class EightPuzzleAStar {
    static int[][] goal = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 0}
    };
    
    static int heuristic(int[][] state) {
        int h = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (state[i][j] != 0) {
                    int val = state[i][j] - 1;
                    int goalX = val / 3, goalY = val % 3;
                    h += Math.abs(i - goalX) + Math.abs(j - goalY);
                }
            }
        }
        return h;
    }
    
    static List<int[][]> getNeighbors(int[][] state) {
        List<int[][]> neighbors = new ArrayList<>();
        int x = 0, y = 0;
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (state[i][j] == 0) {
                    x = i;
                    y = j;
                }
            }
        }
        
        int[][] moves = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] move : moves) {
            int nx = x + move[0], ny = y + move[1];
            if (nx >= 0 && ny >= 0 && nx < 3 && ny < 3) {
                int[][] newState = new int[3][3];
                for (int i = 0; i < 3; i++)
                    newState[i] = state[i].clone();
                
                newState[x][y] = newState[nx][ny];
                newState[nx][ny] = 0;
                neighbors.add(newState);
            }
        }
        return neighbors;
    }
    
    static void solve(int[][] start) {
        PriorityQueue<PuzzleNode> pq = new PriorityQueue<>();
        Set<String> visited = new HashSet<>();
        pq.add(new PuzzleNode(start, 0, heuristic(start), null));
        
        while (!pq.isEmpty()) {
            PuzzleNode current = pq.poll();
            
            if (Arrays.deepEquals(current.state, goal)) {
                System.out.println("Solution found:");
                printSolution(current);
                return;
            }
            
            visited.add(Arrays.deepToString(current.state));
            for (int[][] neighbor : getNeighbors(current.state)) {
                if (!visited.contains(Arrays.deepToString(neighbor))) {
                    pq.add(new PuzzleNode(neighbor, current.g + 1, heuristic(neighbor), current));
                }
            }
        }
        System.out.println("No solution found");
    }
    
    static void printSolution(PuzzleNode node) {
        if (node == null) return;
        printSolution(node.parent);
        System.out.println("Step:");
        for (int[] row : node.state) {
            System.out.println(Arrays.toString(row));
        }
        System.out.println();
    }
    
    public static void main(String[] args) {
        int[][] start = {
            {1, 2, 3},
            {4, 0, 5},
            {7, 8, 6}
        };
        solve(start);
    }
}
`,
  },
  {
    name: "tsp",
    code: `import java.io.*;
import java.util.*;

public class TSP {

    static int N;
    static int[][] graph;

    public static void readGraph(String filename) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        N = Integer.parseInt(br.readLine());  // Number of cities
        graph = new int[N][N];
        
        // Read the cost matrix (graph)
        for (int i = 0; i < N; i++) {
            String[] line = br.readLine().split(" ");
            for (int j = 0; j < N; j++) {
                graph[i][j] = Integer.parseInt(line[j]);
            }
        }
        br.close();
    }

    public static int calculateCost(List<Integer> path) {
        int cost = 0;
        for (int i = 0; i < path.size() - 1; i++) {
            cost += graph[path.get(i)][path.get(i + 1)]; // cost of adjacent nodes
        }
        return cost;
    }

    // BFS for TSP
    public static void bfs(int start) {
        Queue<List<Integer>> queue = new LinkedList<>();
        queue.add(Arrays.asList(start));
        int minCost = Integer.MAX_VALUE;
        List<Integer> bestPath = null;

        while (!queue.isEmpty()) {
            List<Integer> path = queue.poll();

            if (path.size() == N) {
                path.add(start); // add first city
                int cost = calculateCost(path);
                if (cost < minCost) {
                    minCost = cost;
                    bestPath = path;
                }
                continue;
            }

            // Explore all the cities that haven't been visited yet
            for (int i = 0; i < N; i++) {
                if (!path.contains(i)) {
                    List<Integer> newPath = new ArrayList<>(path);
                    newPath.add(i);
                    queue.add(newPath);
                }
            }
        }

        System.out.println("BFS Best Path: " + bestPath + " Cost: " + minCost);
    }

    // DFS for TSP
    public static void dfs(int start) {
        Stack<List<Integer>> stack = new Stack<>();
        stack.push(Arrays.asList(start));
        int minCost = Integer.MAX_VALUE;
        List<Integer> bestPath = null;

        while (!stack.isEmpty()) {
            List<Integer> path = stack.pop();

            if (path.size() == N) {
                path.add(start);  // add start
                int cost = calculateCost(path);
                if (cost < minCost) {
                    minCost = cost;
                    bestPath = path;
                }
                continue;
            }

            for (int i = 0; i < N; i++) {
                if (!path.contains(i)) {
                    List<Integer> newPath = new ArrayList<>(path);
                    newPath.add(i);
                    stack.push(newPath);
                }
            }
        }

        System.out.println("DFS Best Path: " + bestPath + " Cost: " + minCost);
    }

    public static void main(String[] args) throws IOException {
        readGraph("filename.txt");

        // Run BFS and DFS starting from city 0
        bfs(0);
        dfs(0);
    }
}


`,
  },
  {
    name: "tsp_gbfs",
    code: `import java.util.*;

class TSPGreedyBestFirst {
    static int N = 4;
    static int[][] graph = {
        {0, 15, 25, 15},
        {15, 0, 34, 40},
        {25, 34, 0, 28},
        {15, 40, 28, 0}
    };

    static void greedyBestFirstSearch(int start) {
        boolean[] visited = new boolean[N];
        List<Integer> path = new ArrayList<>();
        int totalCost = 0, current = start;
        
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[1]));
        visited[current] = true;
        path.add(current);

        while (path.size() < N) {
            pq.clear();
            
            // add unvisited neighbors to pq
            for (int j = 0; j < N; j++) {
                if (!visited[j]) {
                    pq.add(new int[]{j, graph[current][j]}); // 0 is the node, 1 is the cost
                }
            }

            if (pq.isEmpty()) break; // No more paths left

            // select best node (lowest cost)
            int[] next = pq.poll();
            path.add(next[0]);
            visited[next[0]] = true;
            totalCost += next[1];
            current = next[0]; // Move to next node
        }

        // return to start to complete the cycle
        totalCost += graph[current][start];
        path.add(start);

        System.out.println("Path: " + path);
        System.out.println("Total Cost: " + totalCost);
    }

    public static void main(String[] args) {
        greedyBestFirstSearch(0);
    }
}

`,
  },
  {
    name: "tsp_astar",
    code: `import java.util.*;

class TSPAStar {
    static int N = 4;
    static int[][] graph = {
        {0, 15, 25, 15},
        {15, 0, 34, 40},
        {25, 34, 0, 28},
        {15, 40, 28, 0}
    };

    static class Node implements Comparable<Node> {
        int city, g, h, f;
        List<Integer> path;

        Node(int city, int g, int h, List<Integer> path) {
            this.city = city;
            this.g = g;
            this.h = h;
            this.f = g + h;
            this.path = new ArrayList<>(path);
        }

        public int compareTo(Node other) {
            return Integer.compare(this.f, other.f);
        }
    }

    static int heuristic(int city, Set<Integer> remaining) {
        if (remaining.isEmpty()) return 0;
        return remaining.stream().mapToInt(c -> graph[city][c]).min().orElse(0);
    }

    static void aStarTSP(int start) {
        PriorityQueue<Node> openList = new PriorityQueue<>();
        Set<String> closedList = new HashSet<>();

        openList.add(new Node(start, 0, heuristic(start, new HashSet<>(List.of(1, 2, 3))), Arrays.asList(start)));

        while (!openList.isEmpty()) {
            Node current = openList.poll();
            String key = current.path.toString();

            if (closedList.contains(key)) continue;
            closedList.add(key);

            if (current.path.size() == N) {
                int finalCost = current.g + graph[current.city][start];
                current.path.add(start);
                System.out.println("Path: " + current.path);
                System.out.println("Total Cost: " + finalCost);
                return;
            }

            Set<Integer> remaining = new HashSet<>();
            for (int i = 0; i < N; i++) if (!current.path.contains(i)) remaining.add(i);

            for (int nextCity : remaining) {
                List<Integer> newPath = new ArrayList<>(current.path);
                newPath.add(nextCity);
                int gCost = current.g + graph[current.city][nextCity];
                int hCost = heuristic(nextCity, remaining);
                openList.add(new Node(nextCity, gCost, hCost, newPath));
            }
        }
    }

    public static void main(String[] args) {
        aStarTSP(0);
    }
}

`,
  },
  {
    name: "propositional",
    code: `class Clause:
    def __init__(self, literals):
        self.literals = set(literals)  # Store literals in a set for easy resolution

    def resolve(self, other):
        new_clauses = set()
        for lit in self.literals:
            if ('~' + lit) in other.literals or (lit[1:] if lit.startswith('~') else '~' + lit) in other.literals:
                new_literals = (self.literals | other.literals) - {lit, ('~' + lit) if lit[0] != '~' else lit[1:]}
                new_clauses.add(Clause(new_literals))
        return new_clauses

    def is_empty(self):
        return len(self.literals) == 0

    def __repr__(self):
        return " OR ".join(self.literals) if self.literals else "∅"


class Resolution:
    def __init__(self, knowledge_base):
        self.kb = [Clause(stmt) for stmt in knowledge_base]

    def resolution(self, query):
        negated_query = Clause({'~' + q if q[0] != '~' else q[1:] for q in query})
        clauses = set(self.kb + [negated_query])

        while True:
            new_clauses = set()
            clauses_list = list(clauses)

            for i in range(len(clauses_list)):
                for j in range(i + 1, len(clauses_list)):
                    resolvents = clauses_list[i].resolve(clauses_list[j])

                    if any(res.is_empty() for res in resolvents):
                        return True  # Contradiction found

                    new_clauses.update(resolvents)

            if new_clauses.issubset(clauses):
                return False  # No new clauses, query cannot be resolved

            clauses.update(new_clauses)


# Example Usage
knowledge_base = [
    {"~P", "Q"},  # ¬P ∨ Q
    {"P"},        # P
]

query = {"Q"}  # Check if Q is entailed

resolver = Resolution(knowledge_base)
result = resolver.resolution(query)

print("Query is Entailed" if result else "Query is NOT Entailed")

`,
  },
  {
    name: "predicate",
    code: `class PredicateLogic:
    def __init__(self):
        self.clauses = [] # list of clauses

    def add_clause(self, clause):
        self.clauses.append(clause)

    def unify(self, literal1, literal2):
        if literal1 == literal2:
            return {}
        return None

    def resolve(self, new_clause):
        for clause in self.clauses:
            for literal1 in new_clause:
                for literal2 in clause:
                    if literal1.startswith('¬'):
                        negated_literal = literal1[1:]
                    else:
                        negated_literal = '¬' + literal1

                    substitution = self.unify(negated_literal, literal2)
                    if substitution is not None:
                        resolved_clause = self.apply_substitution(new_clause, clause, substitution)
                        print(f"Resolved clause: {resolved_clause}")
                        return resolved_clause
        return None

    def apply_substitution(self, clause1, clause2, substitution):
        resolved_clause = [lit for lit in clause1 if lit not in substitution]
        resolved_clause += [lit for lit in clause2 if lit not in substitution]
        return resolved_clause

    def __str__(self):
        # Return a string representation of the clauses.
        
        return '\n'.join([' OR '.join(clause) for clause in self.clauses])

logic = PredicateLogic()
logic.add_clause(['P(x)', 'Q(y)'])
logic.add_clause(['¬P(a)', 'R(z)'])
print("Clauses:")
print(logic)
resolved = logic.resolve(['¬Q(y)', 'S(c)'])
print("After resolution:")
print(resolved)
`,
  },
  {name : "linear and logistic reg", 
  code : `import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate or load dataset
X = np.array([[1], [2], [3], [4], [5]])  # Feature (Independent variable)
y = np.array([1, 2, 2.9, 4.1, 5.2])    # Target (Dependent variable)

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 6: Visualize the results
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load the dataset (binary classification)
X = np.array([[1], [2], [3], [4], [5]])   # Feature
y = np.array([0, 0, 0, 1, 1])             # Target (binary)

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Step 6: Visualize the decision boundary
X_vals = np.linspace(X.min() - 1, X.max() + 1, 100).reshape(-1, 1)
y_probs = model.predict_proba(X_vals)[:, 1]

plt.plot(X_vals, y_probs, color='red', label='Logistic Curve')
plt.scatter(X, y, color='blue', label='Data points')
plt.axhline(0.5, color='green', linestyle='--', label='Decision Boundary')
plt.title('Logistic Regression')
plt.xlabel('X')
plt.ylabel('Probability')
plt.legend()
plt.show()

# Print evaluation metrics
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
`,

  },

  {
    name: "findS candidate elim",
    code: `data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']
]

hypothesis = ['Ø'] * 6

for row in data:
    if row[-1] == 'Yes':
        for i in range(6):
            if hypothesis[i] == 'Ø':
                hypothesis[i] = row[i]
            elif hypothesis[i] != row[i]:
                hypothesis[i] = '?'

print("Final Hypothesis:", hypothesis)



import copy

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No']
]

attributes = len(data[0]) - 1
S = data[0][:-1]
G = [['?' for _ in range(attributes)]]

for row in data:
    if row[-1] == 'Yes':
        for i in range(attributes):
            if S[i] != row[i]:
                S[i] = '?'
        G = [g for g in G if all(g[i] == '?' or g[i] == S[i] for i in range(attributes))]
    else:
        G_new = []
        for g in G:
            for i in range(attributes):
                if g[i] == '?':
                    for val in ['Sunny', 'Rainy', 'Warm', 'Cold', 'Normal', 'High', 'Weak', 'Strong', 'Change', 'Same']:
                        if val != row[i]:
                            new_hypo = g[:]
                            new_hypo[i] = val
                            G_new.append(new_hypo)
        G = G_new

print("Final Specific Hypothesis:", S)
print("Final General Hypotheses:", G)


`,
  },
  {
    name:'backprop and dt id3',
    code:`import numpy as np

# Sigmoid function and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Training data: XOR logic
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])  # XOR output

# Seed for reproducibility
np.random.seed(1)

# Initialize weights
input_layer_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Weights and biases
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Training loop
epochs = 10000
lr = 0.1

for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Backpropagation
    error = y - final_output
    d_output = error * sigmoid_deriv(final_output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_deriv(hidden_output)

    # Updating weights and biases
    weights_hidden_output += hidden_output.T.dot(d_output) * lr
    weights_input_hidden += X.T.dot(d_hidden) * lr
    bias_output += np.sum(d_output, axis=0, keepdims=True) * lr
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * lr

# Final output after training
print("Final Output after training:\\n", final_output.round(3))






from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np

# ---------- 1. Encode categorical features as integers ----------
# Outlook: 0=Sunny 1=Overcast 2=Rain ; Temp: 0=Hot 1=Mild 2=Cool
# Humidity: 0=High 1=Normal        ; Wind: 0=Weak 1=Strong
X = np.array([
 [0,0,0,0], [0,0,0,1], [1,0,0,0], [2,1,0,0], [2,2,1,0], [2,2,1,1], [1,2,1,1], 
 [0,1,0,0], [0,2,1,0], [2,1,1,0], [0,1,1,1], [1,1,0,1], [1,0,1,0], [2,1,0,1]
])
y = np.array(['Yes','Yes', 'Yes', 'No', 'No',  'No', 'No',
              'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'No'])

# ---------- 2. Train ID3 decision tree ----------
tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(X, y)

# ---------- 3. Predict for a new day ----------
# Example day: Outlook=Sunny, Temp=Cool, Humidity=High, Wind=Strong -> [0,2,0,1]
print("Play Tennis? ->", tree.predict([[2,2,0,0]])[0])

# ---------- 4. Optional: visualize ----------
plot_tree(tree, feature_names=['Outlook','Temp','Humidity','Wind'],
          class_names=tree.classes_, filled=True); plt.show()




`
  },
  {
    name: "User Input Example",
    code: `# User input example
name = input("Enter your name: ")
age = input("Enter your age: ")

print(f"Hello, {name}! You are {age} years old.")`,
  },
]

export default function TemplatesDropdown({ setCode }: TemplatesDropdownProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" className="flex items-center gap-1 text-gray-400 hover:bg-gray-700 hover:text-white">
          <FileCode className="h-4 w-4" />
          <span>Templates</span>
          <ChevronDown className="h-3 w-3" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-56 bg-[#1a222e] text-white border-gray-700">
        <DropdownMenuLabel>Code Templates</DropdownMenuLabel>
        <DropdownMenuSeparator className="bg-gray-700" />

        <DropdownMenuGroup>
          <DropdownMenuLabel className="text-xs text-gray-400 px-2 py-1">AI Algorithms</DropdownMenuLabel>
          {TEMPLATES.slice(0, 10).map((template, index) => (
            <DropdownMenuItem
              key={index}
              className="cursor-pointer hover:bg-gray-700"
              onClick={() => setCode(template.code)}
            >
              {template.name}
            </DropdownMenuItem>
          ))}
        </DropdownMenuGroup>

        <DropdownMenuSeparator className="bg-gray-700" />

        <DropdownMenuGroup>
          <DropdownMenuLabel className="text-xs text-gray-400 px-2 py-1">Basic Examples</DropdownMenuLabel>
          {TEMPLATES.slice(10).map((template, index) => (
            <DropdownMenuItem
              key={index + 10}
              className="cursor-pointer hover:bg-gray-700"
              onClick={() => setCode(template.code)}
            >
              {template.name}
            </DropdownMenuItem>
          ))}
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
