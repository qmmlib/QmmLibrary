function model = structured_train(training_label_vector, training_instance_matrix, liblinear_options, structure)
%STRUCTURED_TRAIN Train structured linear SVM models
%   model = structured_train(training_label_vector, training_instance_matrix, liblinear_options, structure) returns trained model
%   training_label_vector, training_instance_matrix, liblinear_options parameters are same with LIBLINEAR's MATLAB train function
%   structure is 's' for symmetric weighs, 'a' for antisymmetric weights or a matrix for custom structure

if length(structure) == 1 % symmetric or anti-symmetric
    if structure == 's' % symmetric
        mat = generate_matrix(size(training_instance_matrix,2),0);
    elseif structure == 'a' % anti-symmetric
        mat = generate_matrix(size(training_instance_matrix,2),1);
    end
else % free matrix
    mat = structure;
end

new_training_instance_matrix = [];
for i = 1:size(training_instance_matrix,1)
    new_training_instance_matrix = [new_training_instance_matrix ; [mat'*training_instance_matrix(i,:)']'];
end

clear training_instance_matrix;
new_training_instance_matrix = sparse(new_training_instance_matrix);

model = train(training_label_vector, new_training_instance_matrix, liblinear_options);
model.w = mat*model.w';
model.w = model.w';

end

function mat = generate_matrix(dim,asym)

newdim = ceil(dim/2);
mat1 = eye(newdim);
mat2 = eye(newdim);
mat2 = fliplr(mat2);

if asym == 1
    mat2 = (-1) .* mat2;
end

if mod(dim,2) ~= 0
   mat2(1,:) = [];
end

mat = [mat1 ; mat2];

end
