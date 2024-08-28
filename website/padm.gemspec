# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = "padm"
  spec.version       = "0.1.0"
  spec.authors       = ["Viraj Parimi"]
  spec.email         = ["parimiviraj@gmail.com"]

  spec.summary       = "Write a short summary"
  spec.homepage      = "http://address.com"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select { |f| f.match(%r!^(assets|_layouts|_includes|_sass|LICENSE|README)!i) }

  spec.add_runtime_dependency "jekyll", "~> 3.9"

  spec.add_development_dependency "bundler", "~> 2.4.8"
  spec.add_development_dependency "rake", "~> 12.3.3"
end
