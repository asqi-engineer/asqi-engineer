:layout: landing

ASQI Engineer
=============

**ASQI (AI Solutions Quality Index) Engineer** is a comprehensive framework for systematic testing and quality assurance of AI systems. Developed from `Resaro's <https://resaro.ai/>`_ experience bridging governance, technical and business requirements, ASQI Engineer enables rigorous evaluation of AI systems through containerized test packages, automated assessment, and durable execution workflows.

ASQI Engineer is in active development and we welcome contributors to contribute new test packages, share score cards and test plans, and help define common schemas to meet industry needs. Our initial release focuses on comprehensive chatbot testing with extensible foundations for broader AI system evaluation.

.. container:: buttons

    `Quick Start <quickstart.html>`_
    `GitHub <https://github.com/asqi-engineer/asqi-engineer>`_

Key Features
------------

.. grid:: 1 1 3 3
    :gutter: 2

    .. grid-item-card:: ⚡ Durable Execution
        :text-align: center

        `DBOS <https://github.com/dbos-inc/dbos-transact-py>`_-powered fault tolerance with automatic retry and recovery for reliable test execution.

    .. grid-item-card:: 🐳 Container Isolation
        :text-align: center

        Reproducible testing in isolated Docker environments with consistent, repeatable results.

    .. grid-item-card:: 🎭 Multi-System Orchestration
        :text-align: center

        Coordinate target, simulator, and evaluator systems in complex testing workflows.

    .. grid-item-card:: 📊 Flexible Assessment
        :text-align: center

        Configurable score cards map technical metrics to business-relevant outcomes.

    .. grid-item-card:: 🛡️ Type-Safe Configuration
        :text-align: center

        Pydantic schemas with JSON Schema generation provide IDE integration and validation.

    .. grid-item-card:: 🔄 Modular Workflows
        :text-align: center

        Separate validation, test execution, and evaluation phases for flexible CI/CD integration.


LLM Testing
------------

For our first release, we have introduced the ``llm_api`` system type and contributed five test packages for comprehensive LLM system testing. We have also open-sourced a draft ASQI score card for customer chatbots that provides mappings between technical metrics and business-relevant assessment criteria.

LLM Test Containers
^^^^^^^^^^^^^^^^^^^

- **Garak**: Security vulnerability assessment with 40+ attack vectors and probes
- **DeepTeam**: Red teaming library for adversarial robustness testing  
- **TrustLLM**: Comprehensive framework and benchmarks to evaluate trustworthiness of LLM systems
- **Inspect Evals**: Comprehensive evaluation suite with 80+ tasks across cybersecurity, mathematics, reasoning, knowledge, bias, and safety domains
- **Resaro Chatbot Simulator**: Persona and scenario based conversational testing with multi-turn dialogue simulation

The ``llm_api`` system type uses OpenAI-compatible API interfaces. Through `LiteLLM <https://github.com/BerriAI/litellm>`_ integration, ASQI Engineer provides unified access to 100+ LLM providers including OpenAI, Anthropic, AWS Bedrock, Azure OpenAI, and custom endpoints.

Test Packages
-------------

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: 🔒 Security Testing
        :text-align: center

        Comprehensive vulnerability assessment with **Garak** (40+ security probes) and **DeepTeam** (advanced red teaming) frameworks.

        +++

        .. button-ref:: test-containers
            :ref-type: doc
            :color: primary
            :outline:

            Security Containers

    .. grid-item-card:: 💬 Conversation Quality  
        :text-align: center

        Multi-turn dialogue testing with **persona-based simulation** and **LLM-as-judge evaluation** for realistic chatbot assessment.

        +++

        .. button-ref:: test-containers
            :ref-type: doc
            :color: primary
            :outline:

            Quality Testing

    .. grid-item-card:: 🎯 Trustworthiness
        :text-align: center

        Academic-grade evaluation across **6 trust dimensions** using the **TrustLLM** framework for comprehensive assessment.

        +++

        .. button-ref:: test-containers
            :ref-type: doc
            :color: primary
            :outline:

            Trust Evaluation

    .. grid-item-card:: 🔧 Custom Testing
        :text-align: center

        Build **domain-specific test containers** with standardized interfaces and **multi-system orchestration** capabilities.

        +++

        .. button-ref:: custom-test-containers
            :ref-type: doc
            :color: primary
            :outline:

            Create Containers

Beta: ASQI Chatbot Quality Index
--------------------------------

.. warning::
   🚧 This is a draft quality index under active development.

ASQI Engineer includes a **draft comprehensive quality index specifically designed for chatbot systems**. This beta feature provides a standardized framework for evaluating chatbot quality across multiple dimensions that matter to businesses deploying conversational AI.

What is the ASQI Chatbot Quality Index?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ASQI Chatbot Quality Index is a multi-dimensional assessment framework that evaluates chatbot systems across performance and risk handling across eight key areas:

- **Relevance**: How relevant is the information provided by the chatbot?
- **Accuracy**: How correct is the information provided by the chatbot?
- **Consistency**: How consistently does the chatbot perform when users express the same intent using different words, styles, or structures?
- **Out-of-domain Handling**: How well does the chatbot identify when users are asking for something it's not designed to do?
- **Bias Mitigation**: How effectively does the chatbot avoid biased, stereotypical, or discriminatory responses?
- **Toxicity Control**: To what extent is offensive or toxic output controlled?
- **Competition Mention**: How effectively does the chatbot avoid promoting competitors while maintaining appropriate responses when directly asked about market alternatives?
- **Jailbreaking Resistance**: How strong is the resistance to different jailbreaking techniques?

Running the ASQI Chatbot Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The complete evaluation combines multiple test containers and provides comprehensive scoring:

.. code-block:: bash

   # Download the comprehensive chatbot test suite
   curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/asqi_chatbot_test_suite.yaml
   curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/score_cards/asqi_chatbot_score_card.yaml

   # Run comprehensive chatbot evaluation (tests multiple containers)
   asqi execute \
     -t asqi_chatbot_test_suite.yaml \
     -s demo_systems.yaml \
     -r asqi_chatbot_score_card.yaml \
     -o chatbot_asqi_assessment.json

Beta Status and Collaboration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are actively seeking collaboration from the community to:

- **Refine assessment criteria**: Help define industry-standard thresholds and grading scales
- **Expand test coverage**: Contribute new test scenarios and edge cases  
- **Develop domain-specific indices**: Create specialized quality indices for different chatbot use cases
- **Validate against real deployments**: Share feedback from production chatbot evaluations

We welcome contributions through `GitHub Issues <https://github.com/asqi-engineer/asqi-engineer/issues>`_ to discuss collaboration opportunities or share your experience with the beta ASQI Chatbot Quality Index.

Contributors
------------

.. contributors:: asqi-engineer/asqi-engineer
    :avatars:

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:
   :hidden:

   quickstart
   architecture

.. toctree::
   :maxdepth: 2
   :caption: Configuration:
   :hidden:

   configuration
   test-containers
   llm-test-containers
   custom-test-containers

.. toctree::
   :maxdepth: 2
   :caption: Reference:
   :hidden:

   cli
   library
   examples
   autoapi/index

.. toctree::
   :maxdepth: 1
   :caption: Links:
   :hidden:

   GitHub Repository <https://github.com/asqi-engineer/asqi-engineer>
   Discussions <https://github.com/asqi-engineer/asqi-engineer/discussions>
