java c
Chemistry   125-225:   Machine   Learning   in   Chemistry 
Fall   Quarter,   2024 Homework Assignment #2 - Due:    November    11, 2024.    Turn   in   a   writeup   with   your   responses   to   a   ll   questions   below,   codes,   outputs   (e.g.    graphs,   etc.).    Attach   all   your   Python   files   as   well,   so   we       can run   them.
Problem 1: Linear Regression and Least Squares Fitting In   this   problem,   you   will   derive   and   apply   the   least   squares   method   to   fit   a   straight   line   to   data   from   a   real-world   chemistry   dataset.      This   exercise   will   help   you   understand   the   derivation   of least   squares   fitting,   its   implementation,   and   how   it   can   be   applied   in   a   chemical   context.
Background and Derivations 
The   least   squares   method   is   widely   used   to   fit   models   to   data,   one   of the   simplest   applications   being   to   fit   a   straight   line   y   = mx   + b,   where   m   is   the   slope   and   b   is   the   intercept.
(a) Derivation of the Normal Equations Suppose   we   have   a   dataset   consisting   of   N   observations.      Each   observation   includes   an   independent   variable   xi      and   a   dependent   variable   yi   ,   which   are   related   (approximately   linearly)   by   the   equation   yi    ≈   mxi    + b.    Our   objective   is   to   find   the   slope   m   and   intercept   b   that   best   fit   the   data   in   a   least-   squares   sense,   minimizing   the   sum   of squared   residuals   between   the   actual   y-values   and   the   predicted   values   ˆ(y)   = mx   + b.
Define   the   following:
• Vector y:   The   N-dimensional   vector   of observed   y-values.

• Matrix A:   An   N   ×   2 matrix   containing   the   x-values   of   the   observations   in   the   second   column   and a   column   of   ones   in   the   first   column.   This   setup   allows   us   to   solve   for   both   m   and   b   simultaneously.

• Vector x:   The   vector   containing   our   unknown   parameters,   m   and   b.

The   goal   is   to   find   the   vector x such   that   Ax is   as   close   as   possible   to y in   a   least-squares   sense.   This   leads   to   minimizing   the   squared   error   ||Ax − y||2.
i.    Show   that   the   optimal   solution   for x that   minimizes   the   squared   error   satisfies   the   normal   equa-   tion:
ATAx = AT y.
*Hint:*   Start   by   expanding   ||Ax − y||2 and   then   set   the   gradient   with   respect   to x to   zero.
ii.      Using   the   normal   equation   ATAx =   AT y,   derive   explicit   formulas   for   m   and   b   that   apply   when   fitting   a   line.   Show   that   these   are   equivalent   to:

iii.    Suppose   in   a   chemical   experiment,   x   represents   the   concentration   of   a   reactant   (e.g.,   molarity,   M)   and   y   represents   the   rate   of   reaction   observed   at   each   concentration.    Explain   how   finding   m and   b   could   help you   interpret the   relationship   between   concentration   and   reaction   rate,   possibly   leading   to   insights   about   reaction   kinetics.
(b) Download and Explore the Data We   will   use   the   ESOL    (Delaney)   dataset   from   MoleculeNet,    which   provides   information   on   water   solubility   (log   solubility) of   organic   molecules.   Access   this   dataset   using   the   DeepChem   Python   library as   shown   below.
i.   Install   DeepChem   (if not   already   installed)   by   running:   pip    install      deepchem
ii.      Use   the   following   Python   code   to   load   the   dataset   and   extract   features   and   labels:import deepchem as dc# Load the ESOL datasettasks , datasets , transformers = dc . molnet . load_delaney ()train_dataset , valid_dataset , test_dataset = datasets# Extract features (X) and labels (y) from the training datasetX_train = train_dataset .Xy_train = train_dataset .y
iii.      Display the structure of X   train and y train to   understand what   these   variables   represent   in   this   dataset.
iv.      Answer   the   following   questions:
• What   are   the   features   in   this   dataset?
• Which   feature(s)   would   you   expect   to   correlate   with   solubility?
(c) Implement Linear Regression on the Dataset 
i.      Using      the      LinearRegression      model      from    scikit-learn,    train    a      linear      regression      model      on   X train and   y train.
ii.      Use   the   following   code   to   train   the   model   and   calculate   relevant   parameters   (slope,   intercept,   and RMSE):from sklearn . linear_model import LinearRegressionfrom sklearn . metrics import mean_squared_errorimport numpy as np# Train a simple linear regression model on the datasetlinear_regressor = LinearRegression ()linear_regressor . fit ( X_train , y_train )# Obtain slope ( coefficients ) and interceptslope = linear_regressor . coef_ . flatten ()# Flatten if it ’s a multidimensionalarrayintercept = float ( linear_regressor . intercept_ )# Convert intercept to float# Predict target variable for training datay_train_pred = linear_regressor . predict ( X_train )# Calculate RMSErmse = np . sqrt ( mean_squared_error ( y_train , y_train_pred ))# Print results in the terminalprint (f" Slope (s): { slope }")print (f" Intercept : { intercept }")print (f" RMSE : { rmse }")
iii.      Answer   the   following   questions:
• Explain   the   meaning   of each   parameter   (slope,   intercept)   in   the   context   of solubility.
• What   does   the   RMSE   value   indicate   about   the   model’s   accuracy?
(d) Plot the Fitted Line 
i.      Plot   the   actual   versus   predicted   values   of solubility   to   assess   the   fit   visually.import matplotlib . pyplot as plt# Plot actual vs predicted solubility valuesplt . scatter ( y_train , y_train_pred , label =’Predicted vs Actual ’)plt . plot ([ min ( y_train ) , max( y_train )] , [ min ( y_train ) , max( y_train )] , color =’red ’,linestyle. =’--’, label =’Ideal Fit ’)plt . xlabel (" Actual Solubility ")plt . ylabel (" Predicted Solubility ")plt . legend ()plt . title (" Linear Regression on ESOL Dataset ")plt . show ()
ii.      Answer   the   following   questions:
• How   well   does   the   fitted   line   match   the   data   visually?
•      Are   there   any   potential   outliers   or   deviations?
(e) Assumptions in Least Squares Fitting 
i.    Discuss   the   assumptions   behind   least   squares   fitting.
ii.      Answer   the   following   questions:
• What   assumptions   are   made   about   the   distribution   of noise   in   least   squares   fitting?
• How   might   non-Gaussian   noise   affect   the   accuracy   of   your   linear   model?
Submission 
Submit   a   report   containing:
• Code   for   each   part.
•    Plots and   answers   to   the   questions.
• Interpretation   of results.
Problem 2: Nonlinear Least Squares Fitting on Experimental Chem- istry Data In this problem, you will perform. a   nonlinear   least   squares   fit   on   experimental   chemistry   data   from   the   Free-   Solv   dataset,   a   curated   database   of experimental   and   calculated   hydration   free   energies   for   small   molecules   in   water.         This   exercise   will   introduce   you   to   concepts   in   nonlinear   regression,   data   filtering,   and   model   evaluation.The FreeSolv dataset can be   accessed   at https://github.com/MobleyLab/FreeSolv. You will download   the dataset   and perform. the following   steps to   fit   a   model   that   describes   the   relationship   between   calculated   and   experimental   hydration   free   energies.
1. Data Download and Preparation 
(a)      Download   the   database   .txt   file   from   the   FreeSolv   GitHub   repository,   or   use   Python   code   to   download   it   programmatically.
(b)      Load   the   dataset   using   pandas   in   Python,   skipping   comment   lines   (lines   starting   with   #)   and   specifying   the   delimiter   as   a   semicolon.   Name   the   columns   as   follows:
•    Compound   ID,   SMILES,   IUPAC   Name,   Expt   Free   Energy,   Uncertainty,   Calc   Free   Energy,   DOI,   Notes,   Additional   Column   1,   Additional   Column   2
(c)      Extract   the   Calc    Free    Energy   as   the   predictor   variable   X   and   the   Expt    Free    Energy   as   the   response   variable   y.
2. Outlier Detection and Filtering 
(a)    Remove   any   rows   in   X   or   y   that   contain   NaN   or   infinite   values.
(b)代 写Chemistry 125-225: Machine Learning in Chemistry Homework Assignment #2Python
代做程序编程语言      Filter   out   outliers   in   the   dataset   by   removing   data   points   where   the   Expt    Free    Energy   value   is   more   than   3   standard   deviations   away   from   the   mean.   This   step   ensures   that   extreme   values   do not   unduly   influence   the   fit.
3. Model Definition and Fitting 
(a)      Define   a   quadratic   model   of   the   form.
f(x) = ax2   + bx   +   c 
where   a,   b,   and   c   are   parameters   to   be   determined. 
(b)      Use      scipy   .optimize   .curve fit   to   fit   the   model   to   the   filtered   data   and   extract   the   best-fit   parameters   a,   b,   and   c.
4. Plotting the Results 
(a)      Plot   the   filtered   data   points   (as   a   scatter   plot) and   the   fitted   quadratic   model   (as   a   smooth   curve).
(b)      Label   the   axes   appropriately   as   ”Calculated   Free   Energy”   (x-axis)   and   ”Experimental   Free   En-   ergy”   (y-axis).
(c)      Display   the   fitted   parameters   a,   b,   and   c   in   the   plot   title.
5. Model Evaluation 
(a)    Calculate   the   Root   Mean   Square   Error   (RMSE)   to   evaluate   the   model’s   accuracy.   The   RMSE   is defined   as:

where   yi    are the   actual   experimental values   and   f(xi   )   are the   values   predicted   by   your   quadratic   model.
(b)      Print   the   RMSE   to   assess   the   quality   of your   model   fit.
Questions 
(a)   Why   is   it   important   to   remove   outliers   when   performing   regression?      How   could   outliers   affect   the   quality   of your   model?
(b)    Explain   why   a   quadratic   model   was   chosen   here   instead   of   a   simple   linear   or   exponential   model.    Under what   circumstances   would   each   model   type   be   appropriate?
(c)      The   RMSE   provides   a   measure   of fit   quality.   Would   a   lower   RMSE   always   indicate   a   better   model   in   a   physical   or   chemical   context?   Why   or   why   not?
(d)   What   assumptions   are   implicit   in   least   squares fitting regarding the   distribution of errors   in   the   data?   Discuss   how   violations   of these   assumptions   might   influence   your   fit.
Hints: 
•    Use   the   pandas,   numpy,   and   scipy   libraries   in   Python   to   handle   data   processing,   model   fitting,   and   numerical   calculations.
•      For   plotting,   use matplotlib.pyplot.    The   plot   function   can   be   used   to   draw   the   quadratic   fit,   and   the   scatter   function   is   suitable   for   data   points.
•      Ensure   that   any   non-numeric   values   (e.g., missing   data   or   text) in   X   or   yare   handled   before   performing the   fit.
Problem 3:    Nonlinear    Least    Squares Fit Using Gradient Descent in numpyIn this problem, you will   perform   a   nonlinear   least   squares   fit   by   implementing   a   gradient   descent   algorithm   from   scratch   using   only   numpy.      Unlike   previous   problems   where   you   used   scipy   to   handle   optimization,   here   you   will   manually   compute   the   gradient   vector   of   the   loss   function   and   use   it   in   a   gradient   descent   loop.    This   exercise   will   deepen   your   understanding   of   the   principles   behind   nonlinear   least   squares   fitting   and   gradient-based   optimization.
Objective: Fit   a   quadratic   model   to   a   dataset   using   gradient   descent.    Specifically,   given   a   set   of   data   points   (xi   ,   yi   ),   find   parameters   a,   b,   and   c   that   minimize   the   sum   of   squared   errors:

where
f(x) = ax2   + bx   +   c
is   the   quadratic   model   function.
1. Data Normalization (Scaling) (a)      Normalize   the   predictor   variable   X      and      the   response   variable   y      to   have   zero   mean   and   unit variance:
where µX    and σX      are the   mean   and   standard   deviation   of X,   and   similarly   for   y.    This step   helps   stabilize   gradient   descent   by   preventing   large   gradients.
2. Define the Loss Function and Gradient
(a)      Derive   the   loss   function   L   for   the   nonlinear   least   squares   fit:

where   xi    and   yi    are   the   data   points.
(b)    Compute   the   partial   derivatives   of   L   with   respect   to   each   parameter   a,   b,   and   c.    This   will   form. the   gradient   vector   you   will   use   in   gradient   descent.      You   are   given   the   partial   derivative   with   respect   to   a:

Using   a   similar   approach,   compute   the   partial   derivatives   with   respect   to   b   and   c, ∂b/∂L and ∂c/∂L , respectively.
(c)   Write a Python function compute gradient(X,    y,    a,      b,    c) that takes in the data X andy along with   the   parameters   a, b, and   c, and   returns   the   gradient   vector   as   anumpy   array   [∂L/∂a,∂L/∂b,∂L/∂c].
3. Implement Gradient Descent for Optimization 
(a)   Initialize   the   parameters   a,   b,   and   c   with   some   starting   values   (e.g.,   all   zeros).
(b)    Set   a   small   learning   rate   η   (e.g.,   0.00001)   to   prevent   parameter   values   from   growing   too   quickly during   updates.
(c)    Set   a   convergence   threshold   (e.g.,   1   × 10 −6)   to   determine   when   to   stop   iterating.
(d)   Write   a   loop   that   performs   the   following   steps:
i.    Compute   the   current   loss   L   based   on   the   current   values   of   a,   b,   and   c.
ii.      Use   compute gradient   to   calculate   the   gradient   vector   for   the   current   values   of   a,   b,   and   c.   iii.      Update   the   parameters   using   the   gradient   descent   update   rule:

iv.    Break   out   of   the   loop   if   the   change   in   loss   between   iterations   is   smaller   than   the   convergence threshold.
(e)    After   convergence,   print   the   optimized   values   of   a,   b,   and   c.
4. Convert Parameters Back to Original Scale 
(a)    Since   you   optimized   the   parameters   on   the   normalized   data,   convert   the   parameters   back   to   the original   scale   for   accurate   interpretation.   Use   the   following   transformations:

5. Plotting and Evaluation 
(a)      Plot the   original   data   points   and the   fitted   quadratic   model   using the   de-normalized   parameters.   (b)    Plot   the   loss   over   iterations   to   observe   the   convergence   of gradient   descent.
(c)    Calculate   the   final   loss   and   print   it   to   evaluate   the   fit   quality.
Questions 
(a)   Why   is   it   important   to   remove   outliers   when   performing   regression?      How   could   outliers   affect   the   quality   of your   model?
(b)    Explain   why   a   quadratic   model   was   chosen   here   instead   of   a   simple   linear   or   exponential   model.    Under what   circumstances   would   each   model   type   be   appropriate?
(c)      The   RMSE   provides   a   measure   of fit   quality.   Would   a   lower   RMSE   always   indicate   a   better   model   in   a   physical   or   chemical   context?   Why   or   why   not?
(d)   What   impact   does   the   learning   rate   η   have   on   the   convergence   of   gradient   descent?    What   happens   if η   is   too   large   or   too   small?
(e)      Explain   the   purpose   of   data   normalization   in   gradient   descent.   How   does   normalization   affect   conver-   gence   and   stability?
What to Turn In: 
1.   Your   Python   code   implementing   gradient   descent   and   data   normalization.
2.      A   plot   showing   the   original   data   and   the   fitted   quadratic   curve.
3.      A   plot   of the   loss   over   iterations,   illustrating   the   convergence   behavior.
4.    A   brief explanation   answering   each   of   the   questions   posed   above.
5.    The   final   optimized   parameters   a,   b,   and   c   on   the   original   scale,   along   with   the   final   loss.
Hints: 
•    To   track   convergence,   keep   a   record   of the   previous   loss   and   compare   it   with   the   current   loss   in   each   iteration.
•   If   you   observe   very   large   or   NaN   values   in   your   parameters,   reduce   the   learning   rate   or   check   your gradient   computation.
• Use   numpy   operations   (e.g.,   np   .sum,   np.dot)   to   implement   the   gradient   calculations   efficiently.





         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
