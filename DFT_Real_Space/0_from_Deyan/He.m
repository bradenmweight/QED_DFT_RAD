clear

Ng = 10;
Ng3 = Ng^3;
points = linspace(-4, 4, Ng);

% Construct DVR Kinetic energy operator
nR = length(points);
dx = points(2) - points(1);
index = 0:(Ng-1);
[X1, X2] = meshgrid(index, index);
DiagArray = (pi^2/3).*ones(1,nR);
DiagMat = diag(DiagArray);
XDIFF = X1 - X2;
TMP = X1 - X2;
TMP(1:1+size(TMP,1):end) = 1;
XDIFF_INV = 1./TMP;
XDIFF_INV(1:1+size(XDIFF_INV,1):end) = 1;
OFF = (-1).^XDIFF.*2.*XDIFF_INV.^2;
OFF(1:1+size(OFF,1):end) = 0;
Tii = DiagMat + OFF;
Tii = Tii./dx.^2;
I = eye(length(Tii));
T = kron(kron(Tii, I),I) + kron(kron(I, Tii),I) + kron(kron(I,I),Tii);
%T = T./2;


Zion = 2       % atomic number
Ne = 2         % Number of electrons
a = 0.0311;
b = -0.048;
c = 0.0020;
d = -0.0116;
gamma = -0.1423;
beta1 = 1.0529;
beta2 = 0.3334;

[X,Y,Z] = meshgrid(points, points, points);
X=X(:);
Y=Y(:);
Z=Z(:);
R=sqrt(X.^2 + Y.^2 + Z.^2);
Vext = -(Zion)./R;


n_Gauss = exp(-R.^2/2);
n_Gauss = -Zion*n_Gauss/sum(n_Gauss)/dx^3; % Gaussian compensation charge
V_comp = -Zion./R.*erf(R/sqrt(2));     % Integral of Gaussian comp. charge

e=ones(Ng,1);
Lap_FD_1D = spdiags([e -2*e e], -1:1, Ng, Ng)/dx^2;
I = speye(Ng);
Lap_FD_3D = kron(kron(Lap_FD_1D,I),I) + kron(kron(I,Lap_FD_1D),I)...
    + kron(kron(I,I),Lap_FD_1D);


Vtot = diag(Vext);

for scf_iter = 1:10
  [phi, eps] = eigs(0.5*T + Vtot, Ne/2, 'sa');
  for i = 1:Ne/2
    phi(:,i) = phi(:,i) / norm(phi(:,i));
  end
  phi = phi/dx^(3/2);

  n=0;
  for i = 1:Ne/2
    n = n + 2*phi(:,i).^2;
  end


  Vexch = -(3/pi)^(1/3)*n.^(1/3);
  rs = ( (3/(4*pi))^(1/3) )*(1./n).^(1/3);
  %Vexch = -((9*pi/4)^(1/3))*(3/(4*pi))*(1./rs);
  fx = -((9*pi/4)^(1/3))*(3/(4*pi))*(1./rs);

  fc=zeros(Ng3,1);
  Vcorr=zeros(Ng3,1);

  for i = 1:Ng3
    if rs(i) < 1
       fc(i) = a*log(rs(i))+b+c*rs(i)*log(rs(i))+d*rs(i);
    else
       fc(i) = gamma/(1+beta1*sqrt(rs(i))+beta2*rs(i));
    end
  end

  fxc = fc + fx;

  for i = 1:Ng3
    if rs(i) < 1
       Vcorr(i) = a*log(rs(i))+(b-(a/3))+ ...
                (2/3)*c*rs(i)*log(rs(i))+(1/3)*(2*d-c)*rs(i);
    else
       fc(i) = gamma/(1+beta1*sqrt(rs(i))+beta2*rs(i));
       Vcorr(i) = fc(i)*(1 + (7/6)*beta1*sqrt(rs(i))...
                +(4/3)*beta2*rs(i))/(1+beta1*sqrt(rs(i))+beta2*rs(i));
    end
  end

  Vxc = Vexch + Vcorr;

  Vhart = cgs(Lap_FD_3D, -4*pi*(n+n_Gauss), 1e-7, 1000) - V_comp;
  Vtot = diag(Vxc + Vhart + Vext);
  Ekin = 0;
  for i = 1:Ne/2
    Ekin = Ekin + 2*phi(:,i)'*(0.5*T)*phi(:,i)*dx^3;
  end
  Ekin

  Eext = sum(n.*Vext)*dx^3
  Ehart = 0.5*sum(n.*Vhart)*dx^3
  %Eexch = sum(-(3/4)*(3/pi)^(1/3)*n.^(4/3))*dx^3;
  Exc = sum(fxc.*n)*dx^3
  Etot = Ekin + Eext + Ehart +Exc

end

%eps


