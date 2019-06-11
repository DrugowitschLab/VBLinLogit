%% script to run all vb_{linar,logit}_example_* scripts

example_scripts = {...
    'vb_linear_example.m', ...
    'vb_linear_example_highdim.m', ...
    'vb_linear_example_sparse.m', ...
    'vb_linear_example_modelsel.m', ...
    'vb_logit_example.m', ...
    'vb_logit_example_coeff.m', ...
    'vb_logit_example_highdim.m', ...
    'vb_logit_example_modelsel.m'};

for i = 1:length(example_scripts)
    script_name = example_scripts{i};
    fprintf('Running %s\n', script_name);
    run(script_name);
    if exist('f1','var') == 1
        if isinteger(f1), nf1 = f1; else, nf1 = get(f1,'Number'); end
        if exist('f2','var') == 1
            if isinteger(f2), nf2 = f1; else, nf2 = get(f2,'Number'); end
            fprintf('Figures %d and %d\n\n', nf1, nf2);
        else
            fprintf('Figure %d\n\n', nf1);
        end
    end
    clear('f1','f2');
end
