{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:

  {% block methods %}
  {% if methods %}

   .. automethod:: __init__

  {% if ('__call__' in all_methods) or ('__call__' in inherited_members) %}

   .. automethod:: __call__

 {% endif %}

   .. rubric:: Methods

   .. autosummary::
      :toctree:
   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__mul__', '__getitem__', '__len__'] %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% for item in inherited_members %}
      {%- if item in ['__mul__', '__getitem__', '__len__'] %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
  {% endif %}
  {% endblock %}


  {% block attributes %}
  {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree:
   {% for item in all_attributes %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
  {% endif %}
  {% endblock %}
