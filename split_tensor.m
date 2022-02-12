function [idxx,idxy,idxz]=split_tensor(m,n,d,mb,nb,db)
for i=1:mb+1
    if i==1
        idxx(i)=1;
    end
    if i>1 && i < mb+1
        idxx(i)=idxx(i-1)+floor(m/mb);
    end
    if i==mb+1
        idxx(i)=m+1;
    end
end

for i=1:nb+1
    if i==1
        idxy(i)=1;
    end
    if i>1 && i < nb+1
        idxy(i)=idxy(i-1)+floor(n/nb);
    end
    if i==nb+1
        idxy(i)=n+1;
    end
end

for i=1:db+1
    if i==1
        idxz(i)=1;
    end
    if i>1 && i < db+1
        idxz(i)=idxz(i-1)+floor(d/db);
    end
    if i==db+1
        idxz(i)=d+1;
    end
end
