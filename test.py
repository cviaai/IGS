def train(args, size_x, size_y, device, log_dir, train_generator, val_generator):
    # Load segmentation model and set pattern optimizer
    p_optimizer = get_pattern_optimizer(
        model=get_segmentation_model(args.dataset, f_path=args.model, nchans=args.nchans, nlayers=args.nlayers, device=device),
        freeze_model=args.freeze_model, mode=args.mode, method=args.method, acceleration=args.acceleration,
        imsize=[size_x, size_y], device=device, log_dir=log_dir, comment=args.comment, maxstep=1, lr=args.lr,
        init_scale=args.init_scale, pattern=args.pattern, img_channels=IMG_CHANNELS[args.dataset])
    # Set metrics and loss function generators
    metric_funcs = dict(dice=MetricFuncGen('dice'), ssim=MetricFuncGen('ssim'))
    loss_func = MetricFuncGen(args.loss)
    # Set image verbose freq
    _verbose_train_freq = int(len(train_generator) * 0.5)
    _verbose_val_freq = int(len(val_generator) * 0.33)
    # Set iter num
    if args.method == 'igs':
        # TODO: do for IGS
        niter = int(size_x * size_y / args.acceleration) if args.mode == '2d' else int(size_y / args.acceleration)
        niter -= 4 if args.mode == '2d' else 1
    else:
        niter = args.epoch
    # Run iterations
    if not args.skipval:
        total_steps = niter * (len(train_generator) + len(val_generator))
    else:
        total_steps = niter * len(train_generator)
    pbar = tqdm(total=total_steps)
    for epoch in range(niter):
        for i, data in enumerate(train_generator):
            p_optimizer.update_on_batch(f_func=lambda p: loss_func(y=k_sampler(data, sampling=p),
                                                                   x=data,
                                                                   s_func=p_optimizer.model))
            pbar.set_description(f'training: [{epoch} epoch] [{i}/{len(train_generator)}]')
            pbar.update(1)
            if i % _verbose_train_freq == 0:
                logging.info(f'[Training {epoch} epoch]: batch [{i}/{len(train_generator)}]')
        p_optimizer.update_on_train_end(epoch)
        if not args.skipval:
            for i, data in enumerate(val_generator):
                if (i % _verbose_val_freq == 0) and (epoch % args.val_log_step == 0):
                    p_optimizer.update_on_val_batch(epoch=epoch, batch_idx=i,
                                                    scalar_funcs={
                                                        metric: _FfuncGen(data, p_optimizer.model, k_sampler, f)
                                                        for metric, f in metric_funcs.items()
                                                    },
                                                    image_funcs=dict(image=lambda x: k_sampler(data, sampling=x)['img']),
                                                    verbose_image=True)
                else:
                    p_optimizer.update_on_val_batch(epoch=epoch, batch_idx=i,
                                                    scalar_funcs={
                                                        metric: _FfuncGen(data, p_optimizer.model, k_sampler, f)
                                                        for metric, f in metric_funcs.items()
                                                    },
                                                    image_funcs=dict(image=lambda x: k_sampler(data, sampling=x)['img']),
                                                    verbose_image=False)
                pbar.set_description(f'validation: [{epoch} epoch] [{i}/{len(val_generator)}]')
                pbar.update(1)
                if i % _verbose_val_freq == 0:
                    logging.info(f'[Validation {epoch} epoch]: batch [{i}/{len(val_generator)}]')
        p_optimizer.update_on_epoch(epoch)
    pbar.close()
    p_optimizer.update_on_end(log_dir)