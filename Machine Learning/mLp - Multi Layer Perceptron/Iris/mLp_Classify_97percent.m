K=0;
eek= zeros(104,3);
format long;
I=11;
Q=104;
D = t;
% D = ttest;
Sh1 = [0 zeros(1,q)];
V=0;
Ver=0;
Error=0;
%val = val'
t = t'

New_V
Se = [ones(Q,1) New_V(:,1:4)]

  for k = 1:Q
      Zh1 = Se(k,:) * Wih;				%NAO ESQUECER DO 'K'
      %Sh = [1 tanh(Zh)];
      %Sh = [0.5 (1 - exp(-Zh))./(1 + exp(-Zh))];
      Sh1 = [1 1./(1 + exp(-Zh1))];	
      %$$$$$$$$$$$$
% % %       Zh2 = Sh* Wih2 ;
% % %       %Sh2 = [0.5 tanh(Zh)];
% % %       Sh2 = [0.5 1./(1 + exp(-Zh2))];
      
      %$$$$$$$$$$$$
   
      Yj1 = Sh1 * Whj;						
      Sy1 = 1./(1 + exp(-Yj1));   
       K=K+1;       
     eek(K,1:3) = Sy1;
        
  end
   
  rot_x = t'          % Cj. de treinamento
%rot_x = ttest       %cj. de testes
eek = eek'
%eek = output'
classes = vec2ind(rot_x);
guga = confusionmat(vec2ind(eek), classes)

total_p = trace(guga)/Q;
total = total_p*100

for l=1:104
[a, b] = max(output(l,:))
matriz(l,1) = (b)
end