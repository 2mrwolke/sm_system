import numpy as np

class SM_System():

  '''
  Organizes callables. An SM-System is a system structure of parallel branches.
  Objects can be added.
  '''

  def __init__(self, pre=None, nl=None, post=None, name='sm', is_analytic=False):
    '''
    :param pre (function): First function of branch.
    :param nl (function): Second function of branch.
    :param post (function): Third function of branch.
    :param name (str): Identifies branch.
    :param is_analytic (bool): Adds functionality of 'pre', 'nl', and 'post' not being functions.
    '''
    self.systems = dict({name: dict({'pre': pre,
                                      'nl': nl,
                                      'post': post,
                                    })
                        }
                       )
    self.is_analytic = is_analytic
    pass

  def update(self, subsystem, key, value):
    self.systems[subsystem].update({key: value})

  def call_branch(self, inputs, name='sm'):
    pre  = self.systems[name]['pre']
    nl   = self.systems[name]['nl']
    post = self.systems[name]['post']
    result = post(nl(pre(inputs)))
    return result

  def __str__(self):
    return str(self.systems)

  def __add__(self, sm):
    '''
    Combines two SM_System objects in parallel.
    :param sm (SM_System)
    :return (SM_System)
    '''
    new_system = SM_System()
    new_system.systems.clear()
    new_system.systems.update(self.systems)
    new_system.systems.update(sm.systems)
    return new_system

  def __call__(self, inputs):
    if self.is_analytic:
        result = dict({name: self.call_branch(inputs, name) for name in self.systems})
    if not self.is_analytic:
        result = [self.call_branch(inputs, name) for name in self.systems]
        result = np.sum(np.array(result), axis=0)
    return result