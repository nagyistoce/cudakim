%Adaptive threshoding
function J=AdapThreshold(I)
      
T=0.2;
Th=[T];
a=1;
while (a)
    s1=find(I<=T);
    my1=mean(I(s1));
    s2=find(I>T);
    my2=mean(I(s2));
    T1=(1/2)*(my1+my2);
    if (abs(T1-T)<.0001) a=0; end;
    T=T1;
    Th=[Th,T];
end;

%figure, plot(Th,'rx');title('Threshold values')

%J=T*ones(size(I));

% Use threshold to create BW image
J=zeros(size(I));
r=find(I>T);
J(r)=1;
