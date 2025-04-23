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
    name: "propositional",
    code: `class propLogic:
    def propFun(self, sentence):
        arr = ['~', '+', '.', '->']
        stack = []  
        negate_inside = False  
        i = 0
        while i < len(sentence):
            char = sentence[i]
            
            if char == '~' and i + 1 < len(sentence) and sentence[i + 1] == '(':
                stack.append(True) 
                negate_inside = not negate_inside 
                print('(', end=' ')
                i += 1 
            
            elif char == ')':
                if stack:
                    stack.pop()  
                    negate_inside = not negate_inside  
                print(')', end=' ')
            
            elif char in arr:
                if negate_inside:  
                    if char == '+':
                        print('AND', end=' ')
                    elif char == '.':
                        print('OR', end=' ')
                    else:
                        print({'~': ' ', '->': 'IF'}[char], end=' ')
                else:
                    print({'~': 'NOT', '+': 'OR', '.': 'AND', '->': 'IF'}[char], end=' ')
            
            else:
                if negate_inside:
                    print(f'NOT {char}', end=' ')
                else:
                    print(char, end=' ')
            
            i += 1

obj1 = propLogic()
sentence = input("Enter a sentence: ")
obj1.propFun(sentence)
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
print(resolved)`,
  },
  {
    name: "logistic reg",
    code: `import numpy as np
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
    name: "linear regression",
    code: `import numpy as np
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
`,
  },
  {
    name: "find s",
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
`,
  },
  {
    name: "candidate elimination",
    code: `import copy

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
    name: "water jug",
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
    name: "TSP.JAVA",
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
    name: "tsp_greedy_java",
    code: `import java.util.*;

public class tsp_greedy {
    static int N;
    static int[][] graph;
    
    static void readGraph(int[][] inputGraph) {
        N = inputGraph.length;
        graph = inputGraph;
    }
    
    static void greedyBFS(int start) {
        boolean[] visited = new boolean[N];
        List<Integer> path = new ArrayList<>();
        int cost = 0, current = start;
        
        for (int i = 0; i < N - 1; i++) {
            visited[current] = true;
            path.add(current);
            int nextCity = -1, minCost = Integer.MAX_VALUE;
            
            for (int j = 0; j < N; j++) {
                if (!visited[j] && graph[current][j] < minCost) {
                    minCost = graph[current][j];
                    nextCity = j;
                }
            }
            
            if (nextCity == -1) break;
            cost += minCost;
            current = nextCity;
        }
        
        path.add(start);
        cost += graph[current][start];
        System.out.println("Greedy BFS Path: " + path + " Cost: " + cost);
    }
    
    public static void main(String[] args) {
        int[][] inputGraph = {
            {0, 10, 15, 20},
            {10, 0, 35, 25},
            {15, 35, 0, 30},
            {20, 25, 30, 0}
        };
        
        readGraph(inputGraph);
        greedyBFS(0); // Start from city 0
    }
}
`,
  },
  {name : "tsp_greedy_best_first.java", 
  code : `import java.util.*;

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
`

  },

  {
    name: "tsp_astar_java.",
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
}`,
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
