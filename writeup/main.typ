#import "aiaa_template.typ": *

#show: aiaa_template.with(
  title: "Efficiency Evaluation of Numerical Methods for Approximating Second Order Ordinary Differential Equations.",
  authors: (
    (
      name: "JQX051",
      affiliation: "Higher Level Mathematics Analysis and Approaches",
      job: ""
    ),
  ),
  abstract: []
)

// We generated the example code below so you can see how
// your document will look. Go ahead and replace it with
// your own content!

= Introduction
In writing an extended essay on airfoil efficiency optimization, I used a computational fluid dynamics solver (CFD) to approximate forces exerted on rigid bodies by fluids. Fundamentally, the program solved ordinary and partial differential equations of varying orders among other intermediary processes. Each iteration required considerable computational capacity and time, and thus, finding the most efficient numerical method for solving ordinary differential equations (ODEs) would allow processes like CFD solving to be optimized. The efficiency of a method, as defined for this exploration, is maximized when complexity—measured through runtime—is minimized, while the accuracy of the results is maximized. 

= Numerical Methods

Three approaches to approximating second order ODEs will be evaluated. Namely, the Runge-Kutta midpoint (RK2) method, the multistep predictor-corrector (P-C) method, and the backward Euler method. These methods provide varying approaches to computing solutions differential equations, as will be detailed, which enables the effective comparison of efficiency. 

In describing each method, the second-order ODE describing the simple harmonic motion of a mass-pendulum system,

$ (dif^2 x)/(dif t^2) = -omega^2x, $<shm>

will be used for sample calculations. For all methods, the equation must be decomposed into a system of first-order ODEs: 

$ (dif x)/(dif t) &= v \ 
  (dif^2 x)/(dif t^2) &= (dif v)/(dif t) = -omega^2x $

The samples will consider the initial conditions $omega = sqrt(10)$, $x_0 = 1$, $v_0 = 0$, and $h = 0.01$ where $omega$ is the angular frequency of the system, $x_0$ is the initial displacement of the mass from equilibrium, $v_0$ is the initial velocity of the mass, and $h$ is the iterative step size.

#linebreak()

== Runge-Kutta Midpoint (RK2) Method 

The Runge-Kutta Midpoint (RK2) method is a numerical method for computing 2nd order ODEs based on the higher order RK4 method. In essence, RK2 operates similar to the Euler method, but introduces a half step (midpoint) between iterations to improve the accuracy of the approximations. To find $x_(n+1)$ and $v_(n+1)$ at time $t_(n+1) = t_n + h$, the derivative $k$ of the initial point is used to calculate values at midpoint $m$:

$ k 1_v &= -omega^2x_n \
k 1_x &= v_n \
m_v &= v_n + 1/2h times k 1_v \
m_x &= x_n + 1/2h times k 1_x.
$

The midpoint values are then used to evaluate the final values:

$ k 2_v &= -omega^2m_x \
k 2_x &= m_v \
x_(n+1) &= x_n + h times k 2_x \
v_(n+1) &= v_n + h times k 2_v.
$

Given the prescribed initial conditions, the ODE can be approximated with RK2 as follows, starting with the half step,

$ k 1_v &= -10 times 1 = -10 \
k 1_x &= 0 \
m_v &= 0 + 1/2 times 0.01 times (-10) = -0.05\
m_x &= 1 + 1/2 times 0.01 times 0 = 1,
$

followed by the full step,

$ k 2_v &= -10 times 1 = -10 \
k 2_x &= -0.05 \
x_(n+1) &= 1 + 0.01 times (-0.05) = 0.9995 \
v_(n+1) &= 0 + 0.01 times (-10) = -0.1.
$

To predict the motion of the pendulum system over time, these final values are used to approximate the next iteration. The RK2 method involves 8 main calculations per iteration for 2nd order ODEs, and thus, the number of operations is proportional to the number of iterations. 

#linebreak()

== Multistep Predictor-Corrector Method (P-C)

The predictor-corrector (P-C) method is a multistep explicit method involving a predictor step, in which the values are predicted using the Euler method, then anjusted in the corrector step using the Adams-Bashforth Method. To find $x_(n+1)$ and $v_(n+1)$ at time $t_(n+1) = t_n + h$, the Euler method is applied to predict $v$ and $y$ at time $t_n+1$:

$ v_("prediction") &= v_n + h times -omega^2x_n \
x_("prediction") &= x_n + h times v_n.
$

This is followed by the corrector step, which is applied conditionally. For the first iteration where a previous $v$ value is unavailable, 

$ v_("corrected") &= v_("prediction"). $

Otherwise, for later iterations where a previous $v$ value is available, 

$ v_("corrected") &= v_n + h(3/2(-omega^2x_n) - 1/2(-omega^2x_(n-1))). $

In both cases, $x_("corrected")$ is calculated as

$ x_("corrected") &= x_n + h/2 (v_n + v_("corrected")). $

For the computation of following iterations, the following assignments can be made:

$ x_(n+1) &= x_("corrected") \
v_(n+1) &= v_("corrected"). $

Given the prescribed initial conditions, the ODE can be approximated with P-C as follows, starting with the prediction,

$ v_("prediction") &= 0 + 0.01 times (-10 times 1) = -0.1 \
x_("prediction") &= 1 + 0.01 times 0 = 1,
$

followed by the corrector, 

$ v_(n+1) &= -0.1 " (since it is the first step)" \
x_(n+1) &= 1 + 0.01/2 ( 0 - 0.1) = 0.9995.
$

As with RK2, to predict the motion of the pendulum system over time, these final values are used to approximate the next iteration. The P-C method involves 4 main calculations per iteration for 2nd order ODEs. 

#linebreak()

== Backward Euler Method

The backward Euler method is an implicit method that uses the unknown slope at the next point to approximate the function's value at that point. To find $x_(n+1)$ and $v_(n+1)$ at time $t_(n+1) = t_n + h$, implicit equations are forumlated:

$ x_(n+1) &= x_n + h times v_(n+1) \ 
v_(n+1) &= v_n + h times (-omega^2x_(n+1)).
$

The system of equations is then solved. Although the system of equations is simple in this example, the potential nonlinearity of equations in more complex systems requires the use of numeric solvers, as will be implemented in the test scripts. 

Given the prescribed initial conditions, the ODE can be approximated with the backward Euler method as follows:

$ x_(n+1) &= 1 + 0.01 times v_(n+1) \ 
v_(n+1) &= 0 - 0.01 times 10 times x_(n+1).
$

Using the numeric solver `fsolve` from the `scipy.optimize` Python module yields the following results:

$ x_(n+1) &= 0.9990 \ 
v_(n+1) &= -0.0999.
$

Although this method only involves 2 fundamental calculations per iteration, there is no definite number of operations involved due to the numerical solving step which involves external libraries and its dependency on the complexity of the equations themselves. 


= Complexity Optimization

The complexity of each numerical method can be quantified by measuring their runtimes. Each method was applied to approximate three practical equations over 10 trials: The equation of motion of a simple harmonic oscillator, 

$ (dif^2 x)/(dif t^2) = -omega^2x, \ 
$

the equation of motion of a damped harmonic oscillator,

$ (dif^2 x)/(dif t^2) + b (dif x)/(dif t) + k x = 0, $

and the Van der Pol oscillator equation, 

$ (dif^2 x)/(dif t^2) - mu(1-x^2)(dif x)/(dif t) + x = 0. $

- present data (complete)
- unfortunately, I am still figuring out how to attach media properly to this PDF. I will attach stuff separately to classroom so you can get an idea of what i'm working with. 
- backward euler is extremely sluggish (due to numeric sys. eq. solving)
- two other methods are faster for some eq. than others
- rk2 fastest overall

- present averages + bar graph comparison
- avg overall time for each method


= Accuracy Optimization

- for SHM and damped equations: plot against analytical solution
- plot absolute error 
- global error analysis: RMSE

- analyse accuracy via convergence rate: 
- run method w different stepsizes (magnitudes of 2)
- calculate rmse of each step size
- analyze error reductiona nd calc convergence rate (convergence rate formula)

$ p approx (log (E(h_1)/E(h_2)))/log(h_1/h_2) $

compare convergence rate across methods
higher order methods = higher efficiency

results: 
- for SHM, only RK2 converges, while the other methods increase in abs. in an increasing oscillating pattern for every iteration
- for damped, all solutions converge. RK2 still converges the fastest, but P-C and Reverse Euler vary between test cases.

- visual comparison of graphs (global)

= Evaluation

RK2 is the best overall

Reverse Euler is lowest performing

P-C is quite accurate across the board and can be used for stiffer equations

Defining Efficiency: accuracy/time -> comparison of this figure leads to RK2 being the most efficient. 

Important consideration + extension:
Reverse euler is much more stable, so it performs well with larger step sizes. This was not changed as it was a control, but it greatly affects its performance as its runtime is comparable to that of the other methods at large step sizes of up to h=3, which is 100x more than the tested size. This does not work for the other two methods.

- this may be considered and explored for the final submission, if advised.

Extension: Test step sizes for full optimization.

Note: "optimization" is better done within method classes themselves. This exploration only covers and outlines the differences between each method class. Focusing on optimizing one solution/method will allow for proper modeling of step size vs. time vs. convergence rate for full optimization. "optimization" in this case is more like "evaluation", which was an oversight when this exploration was planned: method types cannot be quantified for proper optimization.  

Thus, the paper title may be changed to match that: Efficiency Evaluation...

note to Examiner: All the data has been collected, and all of the code and graphs have been generated. I will work to add all of these results for the final submission. 

See the full source code at https://github.com/perakasem/hlaa-ia

- contains code for each numerical method: will be added to appendix for final submission. 


#bibliography(full: true, "bib.yml")
