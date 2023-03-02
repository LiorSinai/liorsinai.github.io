# About

Repository  for my personal [website](https://liorsinai.github.io), based on the [Clean Blog Jekyll](https://startbootstrap.com/themes/clean-blog-jekyll/)  by [David Miller](http://davidmiller.io/). 

## Setup

Requires Jekyll. For installation please see [jekyllrb.com/docs/installation/](https://jekyllrb.com/docs/installation/).

## Running

In a cmd run `jekyll serve` and on a web browser navigate to localhost:4000.

To test mobile:
1. Run `jekyll serve --host 0.0.0.0`.
2. On the mobile on a web browser navigate to the serving computer's IP address followed by port 4000 e.g. 192.168.3.54:4000.

The IPv4 address can be found with `ipconfig` on Windows or `ifconfig` on Linux.

### Troubleshooting

If the mobile website doesn't load try disabling the Firewall. If it then does load, check that there are no Firewall rules blocking it. The rule might apply to port 4000 or the Ruby Interpreter.

## Copyright and License

Copyright 2013-2022 Start Bootstrap LLC. Code released under the [MIT](https://github.com/StartBootstrap/startbootstrap-clean-blog-jekyll/blob/gh-pages/LICENSE) license.
