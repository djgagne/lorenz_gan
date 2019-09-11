function y = Vfdot_AR1(Tf, Vf) %Xf_t, paramf, ef_past, k)

% Xfdot_k
% function used in lorenz96
% calculates the time derivative of Xf_k (parametrized tendencies)
% input values are the row vector Xf;
%                   the constants h, c, b, F;
%                the no variables K, J;
%                       the index k;
%            the AR(1) parameters phi, sd;

global paramf ef_past time s1

% stochastic:
K  = paramf(1);
F  = paramf(2);
bs_0 = paramf(3);
bs_1 = paramf(4);
bs_2 = paramf(5);
bs_3 = paramf(6);
phi = paramf(7);
sd  = paramf(8);
dtf = paramf(9);



Xvf_t = Vf(1:K);

Xfdot = zeros(K,1);

% X variables:

if Tf >= (time+dtf)
    ef_stoch = phi*ef_past + sd*((1-phi^2)^0.5)*randn(s1,K,1);
    ef_past = ef_stoch;
    time = time+dtf;
    while Tf-time>dtf,
        time = time+dtf;
    end
else
    ef_stoch = ef_past;
end

%cubic polynomial plus noise.
g_U = bs_0 + bs_1*Xvf_t + bs_2*Xvf_t.^2 + bs_3 * Xvf_t.^3 + ef_stoch;


Xfdot(1) = -Xvf_t(K)*( Xvf_t(K-1) - Xvf_t(2) ) - Xvf_t(1) + F - g_U(1);
Xfdot(2) = -Xvf_t(1)*( Xvf_t(K)   - Xvf_t(3) ) - Xvf_t(2) + F - g_U(2);
for k=3:K-1
    Xfdot(k) = -Xvf_t(k-1)*( Xvf_t(k-2) - Xvf_t(k+1) ) - Xvf_t(k) + F - g_U(k);
end
Xfdot(K) = -Xvf_t(K-1)*( Xvf_t(K-2) - Xvf_t(1) )  - Xvf_t(K) + F - g_U(K);

y = Xfdot;