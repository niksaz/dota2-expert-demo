import tensorflow as tf
import baselines.common.tf_util as U


def build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> (tf.Variable, tf.Variable)
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = make_obs_ph("observation")
        biases_ph = tf.placeholder(tf.float32, (None, num_actions), name="action_biases")
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
        biased_q_values = q_values + biases_ph
        deterministic_actions = tf.argmax(biased_q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
        _act = U.function(inputs=[observations_ph, biases_ph, stochastic_ph, update_eps_ph],
                         outputs=[output_actions, chose_random],
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        def act(ob, biases_ph, stochastic=True, update_eps=-1):
            return _act(ob, biases_ph, stochastic, update_eps)
        return act


def build_train(
        make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0,
        double_q=True, scope="deepq", reuse=None):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = make_obs_ph("obs_t")
        biases_t = tf.placeholder(tf.float32, (None, num_actions), name="biases_t")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = make_obs_ph("obs_tp1")
        biases_tp1 = tf.placeholder(tf.float32, (None, num_actions), name="biases_tp1")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)
        # action-advice potentials of the taken actions
        f_t_selected = tf.reduce_sum(biases_t * tf.one_hot(act_t_ph, num_actions), 1)

        # target q network evaluation
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

        # compute estimate of best possible value starting from state at t + 1
        # note the use of the biased-greedy policy in the action selection
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_using_online_net = q_tp1_using_online_net + biases_tp1
            q_tp1_best_act = tf.argmax(q_tp1_using_online_net, 1)
        else:
            q_tp1_best_act = tf.argmax(q_tp1 + biases_tp1, 1)
        q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_act, num_actions), 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best
        f_tp1_best = tf.reduce_sum(biases_tp1 * tf.one_hot(q_tp1_best_act, num_actions), 1)

        # compute RHS of bellman equation
        q_t_selected_target = (
            rew_t_ph
            + gamma * q_tp1_best_masked
            + gamma * f_tp1_best
            - f_t_selected)

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
            optimize_expr = optimizer.apply_gradients(gradients)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                biases_t,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                biases_tp1,
                done_mask_ph,
                importance_weights_ph,
            ],
            outputs=[
                td_error,
                weighted_error,
            ],
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, train, update_target, {'q_values': q_values, 'q_func_vars': list(sorted(q_func_vars, key=lambda v: v.name))}
