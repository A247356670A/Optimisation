include "globals.mzn";

% ---------------e0
% enum TOPIC = { BMATHS, AMATHS, ENGLISH, CHEM, PHYS };
% array[TOPIC] of int: hours= [2,5,4,3,6];
% int: m = 2; % precedences
% set of int: PREC = 1..m;
% array[PREC] of TOPIC: before= [BMATHS,CHEM];
% array[PREC] of TOPIC: after=  [AMATHS,PHYS];
% set of TOPIC: morning = {AMATHS,CHEM};
% int: D = 5; 
% int: H = 8;
% set of int: DAY = 1..D;
% set of DAY: HALF = {3,4};

% ---------e1
% enum TOPIC = { BMATHS, AMATHS, STATS, CHEM, PHYS, DIGITAL, GRAPH, OR, DISCRETE, PROG };
% array[TOPIC] of int: hours= [2,5,4,3,6,4,5,4,3,4];
% int: m = 4; % precedences
% set of int: PREC = 1..m;
% array[PREC] of TOPIC: before= [BMATHS,BMATHS,CHEM,GRAPH];
% array[PREC] of TOPIC: after=  [AMATHS,STATS,PHYS,DISCRETE];
% set of TOPIC: morning = {CHEM,DIGITAL,OR};
% int: D = 10; 
% int: H = 8;
% set of int: DAY = 1..D;
% set of DAY: HALF = {3,4,7,9};

% ------------e2
% enum TOPIC = { T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16 };
% array[TOPIC] of int: hours= [2,5,4,3,6,4,5,4,3,4,8,7,6,5,2,6];
% int: m = 8; % precedences
% set of int: PREC = 1..m;
% array[PREC] of TOPIC: before= [T1,T1,T4,T5,T7,T10,T12,T16];
% array[PREC] of TOPIC: after=  [T2,T4,T7,T8,T10,T6,T3,T3];
% set of TOPIC: morning = { T11, T6, T15, T16, T5 };
% int: D = 13; 
% int: H = 8;
% set of int: DAY = 1..D;
% set of DAY: HALF = {3,4,7,9,11};

% ------------e3
enum TOPIC = { T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16 };
array[TOPIC] of int: hours= [5,5,5,5,6,5,5,5,5,5,8,7,6,5,5,6];
int: m = 8; % precedences
set of int: PREC = 1..m;
array[PREC] of TOPIC: before= [T1,T1,T4,T5,T7,T10,T12,T16];
array[PREC] of TOPIC: after=  [T2,T4,T7,T8,T10,T6,T3,T3];
set of TOPIC: morning = { T11, T6, T15, T16, T5 };
int: D = 28; 
int: H = 8;
set of int: DAY = 1..D;
set of DAY: HALF = {3,4,7,9,11,13,14,16,17,20,22};

%-------------------------------------------------
int: h2 = H div 2;
int: totalTime = D*H - 1;
TOPIC: last = max(TOPIC);

array[TOPIC] of var 0..totalTime: start;

array[TOPIC] of var 1..length(TOPIC): intTopic;
array[1..length(TOPIC)] of var TOPIC: newT;
array [TOPIC] of var int: newH;

%------------------------constraints-------------------------
constraint all_different(intTopic);  
constraint all_different(newT);

% New order satisfies before/after.
constraint forall (i in PREC)(
  intTopic[before[i]] < intTopic[after[i]]
);
constraint forall (i in TOPIC)(
  newT[intTopic[i]] = TOPIC[i]
);

% Set new hours follows new order
constraint forall (i in 1..length(intTopic))(
  newH[TOPIC[i]] = hours[newT[TOPIC[i]]]
);

% Each item in start is continued.
constraint forall (i in TOPIC where i < last)(
  start[i] + newH[i] <= start[enum_next(i)] /\
  start[last] + newH[last] <= D*H
);

% Items in start not cross each others
constraint forall (i,j in TOPIC where i < j)(
  start[i] + newH[i] <= start[j] \/
  start[j] + newH[j] <= start[i]
);

% For the day with sport session, no topics join the session 
constraint 
forall (i in HALF)(
  forall(j in TOPIC)(
    (start[j] >= (i-1)*H + H) \/
    (start[j]+ newH[j] <= (i-1)*H + h2  )
));

% For the topic at morning, the start point of topic should be in frist half day and no other topic start ahead of it except topic began last day.
constraint forall(i in morning)(
  (start[TOPIC[intTopic[i]]] mod H < min(h2,min(hours)))
);

% Make sure start point mod H is always smaller than end point mod H, then the topic will always remain In same day.
constraint forall (i in TOPIC)(
  start[newT[i]] mod H < (start[newT[i]] + newH[newT[i]] - 1) mod H
);

% To start as late as possible, the first start point should be maximize. To let the morning topic start as early as possible, the distance between morning topics start points to the first hour of the day should be minimum.
solve maximize min(start) - sum( i in morning)(start[TOPIC[intTopic[i]]] mod H);


output ["TOPIC: \(newT[i])\n start at: DAY\(start[i] div H + 1),hour\(start[i] mod (H )), end at: DAY\((start[i] + newH[i] - 1) div H + 1),hour\((start[i] + newH[i] - 1)mod (H ) + 1) \n" | i in TOPIC]

