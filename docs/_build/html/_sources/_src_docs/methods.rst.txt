Density-based methods
===================


1. SIMP
------------------

    .. math:: E = E_{min} + (E_{max} - E_{min}) \rho ^ p

2. RAMP
-----------------

    .. math:: E = \frac{\rho}{(1 + 8. * (1. - \rho)) * E_{max}}

3. code
-----------------  

.. code-block:: python

      if method =='SIMP':
          C = rho_e**3
      else: # for RAMP method
          C = rho_e/(1 + 8. * (1. - rho_e))

.. toctree::
  :maxdepth: 1
  :titlesonly:




