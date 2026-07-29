# so now we are creating a graph
# and the first thing you create is the state 

import os

# 1 way to create state is typed dict(Most Common Approach)
from typing import TypedDict

class State(TypedDict):
    topic : str
    summary : str
    score : int

# 2 way to create state is by using pydantic approach
# it is good at data validation and type checking at runtime

from pydantic import BaseModel , field_validator

class State(BaseModel):
    topic : str
    score : int
    summary : str = " "

    @field_validator
    def score_positive(cls , v):
        if v < 0 :
            raise ValueError("Score must be Positive")


# 3 way to create state is by using pythobn dataclasses
#standard python dataclass but it is used very rarelty 

from dataclasses import dataclass , field

@dataclass
class State :
    topic : str = " "
    summaru : str = " "
    messages : list = field(default_factory=list)

from langgraph.graph import MessagesState

class State(MessagesState):
    # messages field is already included with add_messages reducer
    # just add your extra fields
    user_name: str
    language: str