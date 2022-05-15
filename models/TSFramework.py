import numpy as np


class TSFramework:
    def __init__(self, agent, budget):
        """Implementation of the teacher-student framework.

        Args:
            agent (Agent): teacher agent
            budget (int): teacher's budget
        """

        self.teacher = agent
        self.budget = budget

    def early_advising(self, action_student, state):
        """Implementation of the early adivise method, which consists in providing more help in the initial states.

        Args:
            action_student (int): the action chosen by the student
            state (numpy.array): the student's current state

        Returns:
            int: the action suggested by the teacher agent
        """

        if self.budget > 0:
            self.budget = self.budget - 1
            action_teacher = self.teacher.get_action(state)

        else:
            action_teacher = action_student

        return action_teacher

    def importance_advising(self, action_student, state, threshold):
        """Implementation of the importance adivise method, which consists in saving the budget to provide advice in the most meaningful states.

        Args:
            action_student (int): the action chosen by the student
            state (numpy.array): the student's current state
            threshold (int): threshold value for the state's importance

        Returns:
            int: the action suggested by the teacher agent
        """

        importance = self.get_max(state) - self.get_min(state)

        if (self.budget > 0) and (importance > threshold):
            self.budget = self.budget - 1
            action_teacher = self.teacher.get_action(state)

        else:
            action_teacher = action_student

        return action_teacher

    def mistake_correcting(self, action_student, state, threshold):
        """Implementation of the early adivise method, which consists in analysing the student's action before providing advice.

        Args:
            action_student (int): the action chosen by the student
            state (numpy.array): the student's current state
            threshold (int): threshold value for the state's importance

        Returns:
            int: the action suggested by the teacher agent
        """

        i = self.get_max(state) - self.get_min(state)
        action_teacher = self.teacher.get_action(state)

        if self.budget > 0 and i > threshold and action_student != action_teacher:
            self.budget = self.budget - 1

        else:
            action_teacher = action_student

        return action_teacher

    def get_max(self, state):
        """Gets the maximun Q-Value for the given state

        Args:
            state (numpy.array): current state

        Returns:
            float: maximun Q-Value
        """

        q_value = self.teacher.model.predict(state)

        return np.amax(q_value[0])

    def get_min(self, state):
        """Gets the minimun Q-Value for the given state

        Args:
            state (numpy.array): current state

        Returns:
            float: minimun Q-Value
        """

        q_value = self.teacher.model.predict(state)

        return np.amin(q_value[0])
