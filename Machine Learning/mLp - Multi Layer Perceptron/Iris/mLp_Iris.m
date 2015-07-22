load Iris_dataset
[x,val,test] = dividerand(irisInputs,0.7,0.15,0.15)
%[t,tval,ttest] = dividerand(irisTargets,0.7,0.15,0.15)
load C:\Users\Gustavo\Documents\MATLAB\MLP\Iris\t.mat
load C:\Users\Gustavo\Documents\MATLAB\MLP\Iris\ttest.mat
load C:\Users\Gustavo\Documents\MATLAB\MLP\Iris\tval.mat
eta = 0.01;				% Learning rate
alpha = 0.7;			% Momentum
tol = 1.4;			% Error tolerance
Q = 104;       			% Total no. of the patterns to be input
n = 4; q = 5; p = 3;	% Architecture

V = x';

%% ###############################################################
%  ###########ANTES DE EXECUTAR ESCOLHER A NORMALIZAÇÃO########### 
%  Normalizar - Usar Maximo e minimo;
%  for i = 1:4,
%     mi = min(V(:,i));
%     ma = max(V(:,i));
%     X(:,i) = (V(:,i)-mi)/(ma-mi);
%  end

%   Normalizar - Usando média e Desvio padrão
    % Média
        md = mean(V)
    % Desvio Padrão   
        desv = std(V)
% for i = 1:4, 
%  New_V(:,i) = (V(:,i)-md(1,i))/(desv(1,i));
% end

%   Normalizar - Usando função Sigmoidal
for i =1:4,
New_V(:,i) =[1./(1 + exp(-(V(:,i)-md(1,i))/(desv(1,i))))]
end
%% ##############################################################
pattern = New_V;
%Wih = 2 * rand(n+1,q) - 1;	
Wih = rand(n+1,q)
%Whj = 2 * rand(q+1,p) - 1;	
Whj = rand(q+1,p)
DeltaWih = zeros(n+1,q);	
DeltaWhj = zeros(q+1,p);
DeltaWihOld = zeros(n+1,q);
DeltaWhjOld = zeros(q+1,p);
Si = [ones(Q,1) pattern(:,1:4)];
D1 = t(1,:)';
D2 = t(2,:)';
D3 = t(3,:)';
Sh = [0.5 zeros(1,q)];	
Sy = zeros(1,p);
Sy2 = zeros(1,p);
Sy3 = zeros(1,p);
deltaO = zeros(1,p);	
deltaH = zeros(1,q+1);	
sumerror = 2*tol;		
J=0;
K=0;

epoca = 200;
tmax = Q*epoca;
I=3;
while ( epoca > J) %(sumerror > tol)%% Iterate
   sumerror = 0;
   errok = [];
   for k = 1:Q
      Zh = Si(k,:) * Wih;				
      Sh = [0.5 1./(1 + exp(-Zh))];	
       % SAÍDA 1
      Yj = Sh * Whj(:,1);						
     	 Sy = 1./(1 + exp(-Yj));
       % SAÍDA 2 
          Yj2 = Sh * Whj(:,2);						
     	 Sy2 = 1./(1 + exp(-Yj2));
       % SAÍDA 3
        Yj3 = Sh * Whj(:,3);						
     	 Sy3 = 1./(1 + exp(-Yj3));
       
       
        Ek1 = D1(k) - Sy;
         Ek2 = D2(k) - Sy2;
          Ek3 = D3(k) - Sy3;
          
          Ek =  [Ek1 Ek2 Ek3]
 
      %deltaO = Ek .* (1 - Sy.^2);
      deltaO = Ek .* Sy .* (1 - Sy);% Delta output
 %################################################################
         K=K+1;
      errok(K,1:3) = Ek;
 %################################################################
      for h = 1:q+1
         DeltaWhj(h,:) = deltaO * Sh(h);
      end
      for h = 2:q+1							 % Delta hidden
         deltaH(h) = (deltaO * Whj(h,:)') * Sh(h) * (1 - Sh(h));
      end 
      for i = 1:n+1							 % Delta W: input-hidden
         DeltaWih(i,:) = deltaH(2:q+1) * Si(k,i);
      end
      Wih = Wih + eta * DeltaWih + alpha * DeltaWihOld;
      Whj = Whj + eta * DeltaWhj + alpha * DeltaWhjOld;      
      DeltaWihOld = DeltaWih;				
      DeltaWhjOld = DeltaWhj;
      sumerror = sumerror + sum(Ek.^2); % Compute error
   end    
   J=J+1;
   e(J,1)=sumerror;
   EkError(J,1)= sum(Ek.^2)/Q;
   
    %Coeficiente de Aprendizado
            
            %coef = 1 - J/tmax;
            coef = J/tmax;
            %a= a/(1+(t/tmax));
            eta = 1.0*(0.001/1.0)^coef; % learning rate
            rate(J,1)= eta;
   
   
   sumerror
end

filename = ['C:\Users\Gustavo\Documents\MATLAB\MLP\Iris\Treinamentos\mLp_Iris', int2str(I), '.mat'];
    save(filename)