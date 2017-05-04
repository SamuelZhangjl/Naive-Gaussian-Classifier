function [samples,lables] = traningSamples(dimension,x1,x2,samplesNum,num)
    if mod(num,2)==0
        %number is even
        r1 = randi([1,samplesNum],num/2,1);
        r2 = randi([1,samplesNum],num/2,1);
        samples = zeros(num,dimension);
        for i = 1:num/2
            samples(i,:) = x1(r1(i),:);
        end
        
        for i = 1:num/2
            samples(i+(num/2),:) = x2(r2(i),:);
        end
        
        lable1 = repmat(1,1,num/2);
        lable2 = repmat(2,1,num/2);
        lables = [lable1,lable2];
        
    else
        %number is odd
     	r1 = randi([1,samplesNum],(num+1)/2,1);
    	r2 = randi([1,samplesNum],(num-1)/2,1);
    	samples = zeros(num,dimension);
        
        for i = 1:(num+1)/2
            samples(i,:) = x1(r1(i),:);
        end
            
     	for i = 1:(num-1)/2
            samples(i+((num+1)/2),:) = x2(r2(i),:);
     	end
        lable1 = repmat(1,1,(num+1)/2);
        lable2 = repmat(2,1,(num-1)/2);
        lables = [lable1,lable2];
    end
end