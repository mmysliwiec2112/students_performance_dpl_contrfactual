% model is a spot that I am quite uncertain of, I probably wrongly understood the way that the predictions are made
% with this model
% defining the neural network and the function for which it will return the probabilities
nn(student_score, [X], Y, [0,1]) :: score(X,Y).

% defining the outcome function
outcome(X,X,win).
outcome(X,Y,loss) :- \+outcome(X,Y,win).

% defining the game function
game(X,Outcome) :-
    score(X, P),
    outcome(S1, P, Outcome).


