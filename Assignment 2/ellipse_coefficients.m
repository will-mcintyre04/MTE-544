% 1. Define the data points
points = [
    2.92, -6.01;
    3.40, -7.20;
    4.99, -7.84;
    5.48, -7.04;
    4.20, -5.91
];

x = points(:, 1);
y = points(:, 2);

% 2. Construct the system of linear equations (M * v = k)
M = [x.^2, 2.*x.*y, y.^2, -2.*x, -2.*y];
k = -ones(5, 1);

% 3. Solve for the polynomial coefficients [A; B; C; D; E]
v = M \ k;

A = v(1);
B = v(2);
C = v(3);
D = v(4);
E = v(5);

% Display the coefficients
fprintf('A = %.4f\nB = %.4f\nC = %.4f\nD = %.4f\nE = %.4f\n\n', A, B, C, D, E);

% 4. Compute Q and b
M_ABC = [A, B; B, C];
M_DE = [D; E];

% Calculate alpha
alpha = 1 / ((M_DE' * (M_ABC \ M_DE)) - 1);
alpha

% Calculate Q matrix
Q = alpha * M_ABC;

% Calculate b vector
b = Q \ (alpha * M_DE);

% Display Q and b
disp('Matrix Q:');
disp(Q);
disp('Vector b:');
disp(b);