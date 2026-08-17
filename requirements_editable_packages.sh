pip list -e --format=columns > requirements_editable_packages.md && \
pip list -e --format=freeze > requirements_editable_packages.txt && \
pip list -e --format=json | python3 -m json.tool > requirements_editable_packages.json
