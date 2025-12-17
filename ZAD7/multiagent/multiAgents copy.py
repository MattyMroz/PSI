# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    # -------------------------------------------------------------------------
    # ZADANIE 1: REFLEX AGENT
    # -------------------------------------------------------------------------
    # OPIS DZIAŁANIA:
    # Agent odruchowy (Reflex) patrzy tylko jeden ruch do przodu.
    # Musimy ocenić, czy dany ruch jest dobry.
    # 1. Jeśli ruch prowadzi do jedzenia -> to dobrze (zwiększamy wynik).
    # 2. Jeśli ruch prowadzi blisko ducha -> to bardzo źle (zmniejszamy wynik drastycznie).
    # 3. Jeśli duch jest przestraszony -> to nie jest groźny.
    # 4. Używamy odwrotności dystansu (1/dystans), żeby bliższe jedzenie było "cenniejsze".
    # -------------------------------------------------------------------------
    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.
        """
        # Pobieramy informacje o stanie gry po wykonaniu potencjalnego ruchu
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # ZADANIE 1: REFLEX AGENT

        # 1. Analiza jedzenia
        # Pobieramy wynik gry
        # Lista współrzędnych (x, y) jedzenia

        # Jeśli jest jakieś jedzenie na planszy
        # Obliczamy dłoguśc dla wszyskich kawałków jedzenia
        # Znajdujemy ten najbliższy
        # Odległość zamieniamy na nagrodę - im bliżej, tym większa nagroda (1/dystans)

        score = successorGameState.getScore()
        foodList = newFood.asList()

        if len(foodList) > 0:
            distancesToFood = [manhattanDistance(
                newPos, foodPos) for foodPos in foodList]
            minDistance = min(distancesToFood)
            score += 1.0 / (minDistance + 1)

        # 2. Analiza duchów
        # Sprawdzamy każdego ducha
        # Pobieramy pozycję ducha
        # Obliczamy dystans do ducha
        # Sprawdzamy czy duch jest aktywny (nie jest przestraszony, timer == 0) i czy jest bardzo blisko (dystans <= 1)
        # Jeśli tak, odejmujemy dużą liczbę punktów, żeby Pacman tam nie poszed
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            distanceToGhost = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer == 0 and distanceToGhost <= 1:
                score -= 1000

        # Zwracamy obliczoną ocenę ruchu
        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    # -------------------------------------------------------------------------
    # ZADANIE 2: MINIMAX
    # -------------------------------------------------------------------------
    # OPIS DZIAŁANIA:
    # Algorytm Minimax zakłada, że przeciwnik gra optymalnie (chce nas zabić).
    # Pacman (Agent 0) to MAX - chce najwyższy wynik.
    # Duchy (Agenci > 0) to MIN - chcą najniższy wynik dla Pacmana.
    #
    # Funkcja jest rekurencyjna:
    # - Wywołujemy ją dla każdego agenta po kolei.
    # - Jeśli agent to Pacman -> wybieramy max z wyników następców.
    # - Jeśli agent to Duch -> wybieramy min z wyników następców.
    # - Głębokość (depth) zmniejszamy dopiero, gdy wszyscy (Pacman + Duchy) wykonają ruch.
    # -------------------------------------------------------------------------
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # ZADANIE 2: MINIMAX

        # Pomocnicza funkcja rekurencyjna do obliczania wartości stanu
        def minimax(agentIndex, depth, state):
            # Sprawdzamy warunki końcowe
            # Jeśli gra się skończyła lub osiągnęliśmy głębokość 0
            # Zwracamy ocenę stanu
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            # Obliczamy indeks następnego agenta
            # numAgents = tyle agentów - 0 to Pacman, 1..n to duchy
            numAgents = state.getNumAgents()  # liczba agentów w grze
            # Indeks następnego agenta
            nextAgent = (agentIndex + 1) % numAgents

            # Jeśli następny agent to Pacman, zmniejszamy depth
            # W przeciwnym razie głębokość pozostaje bez zmian
            if nextAgent == 0:
                nextDepth = depth - 1
            else:
                nextDepth = depth

            # Pobieramy możliwe ruchy dla aktualnego agenta
            legalMoves = state.getLegalActions(agentIndex)

            # Generujemy listę wyników dla każdego możliwego ruchu
            # Dla każdego ruchu tworzymy nowy stan (successor) i wywołujemy rekurencyjnie minimax
            results = []
            for action in legalMoves:
                successor = state.generateSuccessor(agentIndex, action)
                value = minimax(nextAgent, nextDepth, successor)
                results.append(value)

            # JEŚLI TO PACMAN (MAX):
            # Jeśli nie ma ruchów, zwracamy ocenę stanu
            # Pacman wybiera najlepszy (największy) wynik
            # JEŚLI TO DUCH (MIN):
            # Duch wybiera najgorszy dla Pacmana (najmniejszy) wynik
            # Jeśli nie ma ruchów, zwracamy ocenę stanu
            if agentIndex == 0:
                if not results:
                    return self.evaluationFunction(state)
                return max(results)
            else:
                if not results:
                    return self.evaluationFunction(state)
                return min(results)

        # GŁÓWNA CZĘŚĆ METODY
        # Najlepszy wynik zaczyna się od minus nieskończoności
        bestScore = float("-inf")
        bestAction = Directions.STOP  # Domyślna akcja

        # Sprawdzamy każdy legalny ruch Pacmana (Agent 0)
        # Generujemy stan po tym ruchu
        # Obliczamy wartość tego stanu używając naszej funkcji minimax
        # Zaczynamy od agenta 1 (pierwszy duch), głębokość self.depth
        # - Jeśli ten wynik jest lepszy od dotychczasowego najlepszego
        # Aktualizujemy najlepszy wynik
        # Zapamiętujemy akcję, która do niego prowadzi
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)

            score = minimax(1, self.depth, successorState)

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    # -------------------------------------------------------------------------
    # ZADANIE 3: ALPHA-BETA PRUNING
    # -------------------------------------------------------------------------
    # OPIS DZIAŁANIA:
    # To ulepszony Minimax. Działa tak samo, ale przestaje sprawdzać gałęzie,
    # które na pewno nie zostaną wybrane.
    # Alpha: Najlepszy wynik, jaki MAX (Pacman) ma już zagwarantowany.
    # Beta: Najlepszy wynik, jaki MIN (Duch) ma już zagwarantowany.
    #
    # Przycinanie (Pruning):
    # - W węźle MIN: jeśli znajdziemy wartość < alpha, przerywamy (MAX tego nie wybierze).
    # - W węźle MAX: jeśli znajdziemy wartość > beta, przerywamy (MIN na to nie pozwoli).
    # -------------------------------------------------------------------------

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # ZADANIE 3: ALPHA-BETA PRUNING
        # Minimaxie, ale dodajemy tylko parametry alpha i beta oraz warunki przerwania pętli

        def alphabeta(agentIndex, depth, state, alpha, beta):
            # Warunki końcowe
            # Jeśli gra skończona lub głębokość 0 -> zwracamy ocenę
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            # Przygotowanie następnego agenta
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents

            # Jeśli wracamy do Pacmana (0), zmniejszamy głębokość
            if nextAgent == 0:
                nextDepth = depth - 1
            else:
                nextDepth = depth

            # Pobieramy możliwe ruchy
            legalMoves = state.getLegalActions(agentIndex)

            # JEŚLI TO PACMAN (MAX)
            if agentIndex == 0:
                value = float("-inf")  # Szukamy jak największej liczby

                for action in legalMoves:
                    successor = state.generateSuccessor(agentIndex, action)
                    # Rekurencja: przekazujemy alpha i beta dalej
                    currentVal = alphabeta(
                        nextAgent, nextDepth, successor, alpha, beta)

                    # Wybieramy większą wartość
                    if currentVal > value:
                        value = currentVal

                    # PRZYCINANIE (PRUNING):
                    # Jeśli znaleziona wartość jest większa od beta (najlepszej opcji MIN-a),
                    # to MIN (rodzic) nigdy nie wybierze tej ścieżki. Przerywamy!
                    if value > beta:
                        return value

                    # Aktualizujemy alpha (najlepszy wynik jaki MAX może zagwarantować)
                    if value > alpha:
                        alpha = value

                return value

            # JEŚLI TO DUCH (MIN)
            else:
                value = float("inf")  # Szukamy jak najmniejszej liczby

                for action in legalMoves:
                    successor = state.generateSuccessor(agentIndex, action)
                    # Rekurencja
                    currentVal = alphabeta(
                        nextAgent, nextDepth, successor, alpha, beta)

                    # Wybieramy mniejszą wartość
                    if currentVal < value:
                        value = currentVal

                    # PRZYCINANIE (PRUNING):
                    # Jeśli znaleziona wartość jest mniejsza od alpha (najlepszej opcji MAX-a),
                    # to MAX (rodzic) nigdy nie wybierze tej ścieżki. Przerywamy!
                    if value < alpha:
                        return value

                    # Aktualizujemy beta (najlepszy wynik jaki MIN może wymusić)
                    if value < beta:
                        beta = value

                return value

        # GŁÓWNA CZĘŚĆ METODY
        # Musimy wybrać konkretną AKCJĘ

        # Inicjalizacja wartości początkowych
        alpha = float("-inf")
        beta = float("inf")
        bestScore = float("-inf")
        bestAction = Directions.STOP

        # Sprawdzamy ruchy Pacmana w korzeniu
        for action in gameState.getLegalActions(0):
            # Generujemy stan po tym ruchu
            successor = gameState.generateSuccessor(0, action)
            # Wywołujemy funkcję pomocniczą dla pierwszego ducha
            score = alphabeta(1, self.depth, successor, alpha, beta)

            # Szukamy najlepszego wyniku
            if score > bestScore:
                bestScore = score
                bestAction = action

            # Aktualizujemy alpha w korzeniu, żeby kolejne gałęzie wiedziały,
            # że mamy już jakiś dobry wynik (bestScore)
            if bestScore > alpha:
                alpha = bestScore

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    # -------------------------------------------------------------------------
    # ZADANIE 4: EXPECTIMAX
    # -------------------------------------------------------------------------
    # OPIS DZIAŁANIA:
    # Minimax zakłada, że duchy grają idealnie (chcą nas zabić).
    # Expectimax zakłada, że duchy mogą popełniać błędy (grają losowo).
    #
    # Różnica w kodzie:
    # - Pacman (MAX) działa tak samo (chce max).
    # - Duchy (EXP) zamiast brać MIN, liczą ŚREDNIĄ (wartość oczekiwaną).
    # - Wartość = suma wartości wszystkich następców / liczba ruchów.
    # -------------------------------------------------------------------------

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # ZADANIE 4: EXPECTIMAX

        def expectimax(agentIndex, depth, state):
            # Warunki końcowe
            # Jeśli gra skończona lub głębokość 0 -> zwracamy ocenę
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            # Przygotowanie następnego agenta
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents

            # Jeśli wracamy do Pacmana (0), zmniejszamy głębokość
            if nextAgent == 0:
                nextDepth = depth - 1
            else:
                nextDepth = depth

            # Pobieramy możliwe ruchy
            legalMoves = state.getLegalActions(agentIndex)

            # Jeśli brak ruchów, zwracamy ocenę stanu
            if not legalMoves:
                return self.evaluationFunction(state)

            # JEŚLI TO PACMAN (MAX)
            # Działa tak samo jak w Minimax - wybiera najlepszy ruch
            if agentIndex == 0:
                currentMax = float("-inf")

                # Sprawdzamy każdy legalny ruch
                # Generujemy stan po tym ruchu
                # Wywołujemy rekurencyjnie expectimax dla następnego agenta
                for action in legalMoves:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = expectimax(nextAgent, nextDepth, successor)

                    # Szukamy maksimum
                    if value > currentMax:
                        currentMax = value

                return currentMax

            # JEŚLI TO DUCH (EXP - WARTOŚĆ OCZEKIWANA)
            # Liczymy średnią ze wszystkich wyników
            else:
                totalScore = 0.0

                for action in legalMoves:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = expectimax(nextAgent, nextDepth, successor)
                    totalScore += value

                # Zwracamy średnią (suma / ilość ruchów)
                return totalScore / len(legalMoves)

        # GŁÓWNA CZĘŚĆ METODY
        # Musimy wybrać konkretną AKCJĘ dla Pacmana

        bestScore = float("-inf")
        bestAction = Directions.STOP

        # Sprawdzamy ruchy Pacmana w korzeniu
        for action in gameState.getLegalActions(0):
            # Generujemy stan po tym ruchu
            successor = gameState.generateSuccessor(0, action)
            # Wywołujemy funkcję pomocniczą dla pierwszego ducha
            score = expectimax(1, self.depth, successor)

            # Szukamy najlepszego wyniku
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    """
    "*** YOUR CODE HERE ***"
    # ZADANIE 5: LEPSZA FUNKCJA OCENY
    # -------------------------------------------------------------------------
    # ZADANIE 5: LEPSZA FUNKCJA OCENY (WERSJA PROSTA)
    # -------------------------------------------------------------------------
    # OPIS DZIAŁANIA:
    # 1. Baza to aktualny wynik (score).
    # 2. Szukamy najbliższego jedzenia - im bliżej, tym lepiej (dodajemy punkty).
    # 3. Sprawdzamy duchy:
    #    - Jak duch jest normalny i blisko -> OGROMNA KARA (żeby nie zginąć).
    #    - Jak duch jest przestraszony -> mała nagroda (fajnie jak jest blisko).
    # -------------------------------------------------------------------------

    # Pobieramy pozycję Pacmana i aktualny wynik listy jedzenia i duchów
    pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()

    # 1. JEDZENIE
    # Jeśli jest jedzenie, znajdź to najbliższe
    if len(foodList) > 0:
        minDist = min([manhattanDistance(pos, food) for food in foodList])
        score += 10.0 / (minDist + 1)

    # 2. DUCHY
    for ghost in ghostStates:
        dist = manhattanDistance(pos, ghost.getPosition())

        # Jeśli duch jest przestraszony to zachęcamy do podejścia
        if ghost.scaredTimer > 0:
            score += 100.0 / (dist + 1)

        # Jeśli duch jest normalny
        else:
            # Jeśli jest mniej niż 2 pola
            if dist < 2:
                score -= 10000.0  # Duża kara
            else:
                score -= 10.0 / (dist + 1)  # Małą kara by trzymać dystans

    return score


# Abbreviation
better = betterEvaluationFunction
