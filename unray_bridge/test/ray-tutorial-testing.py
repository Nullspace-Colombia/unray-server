import ray

context = ray.init()
print(context.dashboard_url)