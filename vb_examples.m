%% script tu run all vb_linar_example_* scripts

example_scripts = {...
    'vb_linear_example_highdim.m', ...
    'vb_linear_example_sparse.m', ...
    'vb_linear_example_modelsel.m'};

for i = 1:length(example_scripts)
    script_name = example_scripts{i};
    fprintf('Running %s\n', script_name);
    run(script_name);
    fprintf('Figures %d and %d\n\n', f1, f2);
end
